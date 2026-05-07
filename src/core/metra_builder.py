import copy
import os
import numpy as np
import torch
import torch.nn as nn

from utils import utils
from utils.utils import _finalize_lr, OptimizerGroupWrapper
from models import (
    GaussianMLPIndependentStdModuleEx, StopGradEncoder, WithEncoder, PolicyEx,
    ContinuousMLPQFunctionEx, get_gaussian_module_construction
)
from core.encoder_factory import EncoderFactory
from core.actor_critic_factory import ActorCriticModuleFactory, SimBaActorCriticHP
from core.cascade_actor import CascadeActor, CascadeStage, CascadeStageFactory
from core.phi_normalization import build_traj_latent_normalizer
from core.sac_trainer import SacTrainer
from legacy.load_pretrain_metra import load_pretrained_metra
from core.isolation_kernel import SoftIsolationKernel
from core.hierarchical_phi import resolve_total_phi_dim
from core.metra_variants import VariantFactory
from core.stage_contract import (
    effective_skill_dim,
    is_finetune_stage,
    is_zero_training_stage,
    should_build_skill_dynamics,
    should_build_traj_encoder,
    should_use_kme,
)
from utils import agent_utils

class MetraAgentBuilder:
    def __init__(self, config, env, replay_buffer):
        self.cfg = config
        self._env = env
        self.replay_buffer = replay_buffer
        self.device = torch.device(config.device)
        
        # Initialize containers
        self.optimizers_dict = {}
        self.param_modules = {}
        self.components = {}

    def build(self):
        # 1. Setup Encoder & Obs Dimensions
        example_ob = self._env.reset()
        pixel_shape = self._env.spec.observation_space.shape if self.cfg.net.encoder else None
        
        self._init_encoder_and_obs_dims(example_ob, pixel_shape)

        # 2. Setup Modules (Traj Encoder & SAC)
        self._init_modules()

        # 3. Setup Optimizers
        optimizer = OptimizerGroupWrapper(
            optimizers=self.optimizers_dict,
            max_optimization_epochs=None,
        )

        # 4. KME / IDK
        kme_components = self._init_kme_module()
        
        # 5. Algorithm Variant
        variant = VariantFactory.create(self.cfg)

        return {
            'config': self.cfg,
            'env': self._env,
            'replay_buffer': self.replay_buffer,
            'device': self.device,
            'shared_encoder': self.shared_encoder,
            'module_obs_dim': self.module_obs_dim,
            'make_encoder_fn': self.make_encoder_fn,
            'with_encoder_fn': self.with_encoder_fn,
            'traj_encoder': self.traj_encoder,
            'target_traj_encoder': self.target_traj_encoder,
            'traj_latent_normalizer': self.traj_latent_normalizer,
            'target_traj_latent_normalizer': self.target_traj_latent_normalizer,
            'dist_predictor': self.dist_predictor,
            'dual_lam': self.dual_lam,
            'skill_dynamics': self.skill_dynamics,
            'sd_input_bn': self.sd_input_bn,
            'sd_target_bn': self.sd_target_bn,
            'sac_trainer': self.sac_trainer,
            'optimizer': optimizer,
            'param_modules': self.param_modules,
            'variant': variant,
            **kme_components
        }

    def _init_skill_dynamics(self, master_dims, nonlinearity):
        obs_dim = self.module_obs_dim
        if self.traj_encoder:
            obs_dim = self.cfg.algo.dim_skill

        module_cls, module_kwargs = get_gaussian_module_construction(
            hidden_sizes=master_dims,
            const_std=self.cfg.net.sd_const_std,
            hidden_nonlinearity=nonlinearity or torch.relu,
            input_dim=obs_dim + self.cfg.algo.dim_skill,
            output_dim=obs_dim,
            min_std=0.3,
            max_std=10.0,
        )
        self.skill_dynamics = module_cls(**module_kwargs).to(self.device)
        
        if self.cfg.net.sd_batch_norm:
            self.sd_input_bn = torch.nn.BatchNorm1d(obs_dim, momentum=0.01).to(self.device)
            self.sd_target_bn = torch.nn.BatchNorm1d(obs_dim, momentum=0.01, affine=False).to(self.device)
            self.sd_input_bn.eval()
            self.sd_target_bn.eval()
            
        self.param_modules['skill_dynamics'] = self.skill_dynamics
        
        # Optimizer
        self.optimizers_dict['skill_dynamics'] = torch.optim.Adam(
            self.skill_dynamics.parameters(), 
            lr=_finalize_lr(self.cfg.train.lr_te or self.cfg.train.common_lr)
        )

    def _init_encoder_and_obs_dims(self, example_ob, pixel_shape):
        if self.cfg.net.encoder: # for pixels input
            self.shared_encoder = EncoderFactory.create(
                encoder_type=self.cfg.net.encoder_type,
                pixel_shape=pixel_shape,
                finetune=self.cfg.net.finetune_encoder,
                spectral_normalization=self.cfg.net.spectral_normalization,
                device=self.device
            )
            self.shared_encoder.to(self.device)

            def make_encoder(**kwargs):
                return self.shared_encoder

            def with_encoder(module, encoder=None):
                if encoder is None:
                    encoder = self.shared_encoder
                return WithEncoder(encoder=encoder, module=module)

            # Dummy forward to get dim
            dummy_obs = torch.as_tensor(example_ob["image"]).to(self.device).float().unsqueeze(0)
            self.module_obs_dim = self.shared_encoder(dummy_obs).shape[-1]
            self.make_encoder_fn = make_encoder
            self.with_encoder_fn = with_encoder
        else:
            # 1. Extract state vector
            state = agent_utils.extract_state_from_obs(example_ob["info"]['state'])
            state = np.asarray(state, dtype=np.float32)
            self.module_obs_dim = int(np.prod(state.shape))

            # 2. Identity wrappers
            self.shared_encoder = None
            self.make_encoder_fn = None
            self.with_encoder_fn = None

    def _init_modules(self):
        skill_input_dim = effective_skill_dim(self.cfg)
        if skill_input_dim > 0 and self.cfg.algo.use_hierarchical_skill:
            skill_input_dim *= self.cfg.algo.num_skill_levels
        policy_q_input_dim = self.module_obs_dim + skill_input_dim
        action_dim = self._env.spec.action_space.flat_dim
        master_dims = [self.cfg.net.model_master_dim] * self.cfg.net.model_master_num_layers
        
        # Nonlinearity
        if self.cfg.net.model_master_nonlinearity == 'relu':
            nonlinearity = torch.relu
        elif self.cfg.net.model_master_nonlinearity == 'tanh':
            nonlinearity = torch.tanh
        else:
            nonlinearity = None

        # --- Trajectory Encoder ---
        self.traj_encoder = None
        self.target_traj_encoder = None
        self.traj_latent_normalizer = None
        self.target_traj_latent_normalizer = None
        self.dist_predictor = None
        self.dual_lam = None
        
        # --- Skill Dynamics (DADS) ---
        self.skill_dynamics = None
        self.sd_input_bn = None
        self.sd_target_bn = None

        if should_build_traj_encoder(self.cfg):
            self._init_traj_encoder(
                self.make_encoder_fn, self.with_encoder_fn, self.module_obs_dim, master_dims, nonlinearity
            )

        if should_build_skill_dynamics(self.cfg):
            self._init_skill_dynamics(master_dims, nonlinearity)

        # --- SAC ---
        target_entropy = -np.prod(self._env.spec.action_space.shape).item() * self.cfg.train.sac_target_coef
        self._init_sac(
            target_entropy, self.with_encoder_fn, policy_q_input_dim, action_dim, master_dims, nonlinearity
        )

    def _init_traj_encoder(self, make_encoder, with_encoder, module_obs_dim, master_dims, nonlinearity):
        traj_output_dim = resolve_total_phi_dim(self.cfg)
        if traj_output_dim <= 0:
            raise ValueError(f"[METRA] pre_training requires dim_skill > 0.")

        # --- traj head ---
        traj_head = GaussianMLPIndependentStdModuleEx(
            input_dim=module_obs_dim,
            output_dim=traj_output_dim,
            std_hidden_sizes=master_dims,
            std_hidden_nonlinearity=nonlinearity or torch.relu,
            std_hidden_w_init=torch.nn.init.xavier_uniform_,
            std_output_w_init=torch.nn.init.xavier_uniform_,
            init_std=1.0,
            min_std=1e-6,
            max_std=None,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            hidden_w_init=torch.nn.init.xavier_uniform_,
            output_w_init=torch.nn.init.xavier_uniform_,
            std_parameterization='exp',
            bias=True,
            spectral_normalization=self.cfg.net.spectral_normalization,
        ).to(self.device)

        # --- stop-grad shared encoder wrapper ---
        if self.cfg.net.encoder:
            base_encoder = make_encoder()
            stop_grad_encoder = StopGradEncoder(base_encoder).to(self.device)
            self.traj_encoder = WithEncoder(encoder=stop_grad_encoder, module=traj_head).to(self.device)
        else:
            self.traj_encoder = traj_head
        self.traj_encoder.eval()

        # --- target traj encoder ---
        if self.cfg.algo.use_target_traj_encoder:
            if isinstance(self.traj_encoder, WithEncoder):
                self.target_traj_encoder = WithEncoder(
                    encoder=self.traj_encoder.encoder,
                    module=copy.deepcopy(self.traj_encoder.module),
                ).to(self.device)
                for p in self.target_traj_encoder.module.parameters():
                    p.requires_grad_(False)
            else:
                self.target_traj_encoder = copy.deepcopy(self.traj_encoder).to(self.device)
                for p in self.target_traj_encoder.parameters():
                    p.requires_grad_(False)
            self.target_traj_encoder.eval()

        self.traj_latent_normalizer = build_traj_latent_normalizer(
            self.cfg.algo.traj_latent_norm,
            traj_output_dim,
            self.cfg.algo.traj_latent_norm_eps,
        )
        if self.traj_latent_normalizer is not None:
            self.traj_latent_normalizer = self.traj_latent_normalizer.to(self.device)
            self.traj_latent_normalizer.eval()

        if self.cfg.algo.use_target_traj_encoder and self.traj_latent_normalizer is not None:
            self.target_traj_latent_normalizer = copy.deepcopy(self.traj_latent_normalizer).to(self.device)
            for p in self.target_traj_latent_normalizer.parameters():
                p.requires_grad_(False)
            self.target_traj_latent_normalizer.eval()

        # --- dual lambda ---
        self.dual_lam = utils.ParameterModule(
            torch.tensor([np.log(float(self.cfg.algo.dual_lam))], dtype=torch.float32)
        ).to(self.device)

        self.param_modules.update({
            'traj_encoder': self.traj_encoder,
            'dual_lam': self.dual_lam,
        })
        if self.traj_latent_normalizer is not None:
            self.param_modules['traj_latent_normalizer'] = self.traj_latent_normalizer

        # --- optimizer ---
        if self.cfg.net.encoder and isinstance(self.traj_encoder, WithEncoder):
             te_params = list(self.traj_encoder.module.parameters())
        else:
             te_params = list(self.traj_encoder.parameters())
        if self.traj_latent_normalizer is not None:
            te_params = te_params + list(self.traj_latent_normalizer.parameters())

        self.optimizers_dict.update({
            'traj_encoder': torch.optim.Adam([{'params': te_params, 'lr': _finalize_lr(self.cfg.train.lr_te)}]),
            'dual_lam': torch.optim.Adam([{'params': self.dual_lam.parameters(), 'lr': _finalize_lr(self.cfg.train.dual_lr)}]),
        })

    def _init_sac(self, target_entropy, with_encoder, policy_q_input_dim, action_dim, master_dims, nonlinearity):
        # Create Critics
        critic1 = self._create_sac_critic(policy_q_input_dim, action_dim, master_dims, nonlinearity)
        critic2 = self._create_sac_critic(policy_q_input_dim, action_dim, master_dims, nonlinearity)

        # Handle encoder finetuning state
        if self.cfg.net.encoder and not is_finetune_stage(self.cfg):
             if hasattr(self.shared_encoder, "set_finetune"):
                self.shared_encoder.set_finetune(self.cfg.net.finetune_encoder)
             else:
                for p in self.shared_encoder.parameters():
                    p.requires_grad_(self.cfg.net.finetune_encoder)

        # Create Actor
        if not is_finetune_stage(self.cfg):
            skill_policy = self._create_sac_actor(
                with_encoder, policy_q_input_dim, action_dim, master_dims, nonlinearity
            )
            if self.cfg.net.encoder:
                critic1 = with_encoder(critic1)
                critic2 = with_encoder(critic2)
        
        elif is_finetune_stage(self.cfg):
            skill_policy = self._load_actor_from_pretraining().to(self.device)
            skill_policy.train()
            
            if self.cfg.net.encoder:
                if not hasattr(skill_policy, "_module") or not hasattr(skill_policy._module, "encoder"):
                    raise AttributeError("[finetune] loaded skill_policy does not expose _module.encoder.")
                shared_enc = skill_policy._module.encoder
                
                # Set finetune state
                if hasattr(shared_enc, "set_finetune"):
                    shared_enc.set_finetune(self.cfg.net.finetune_encoder)
                else:
                    for p in shared_enc.parameters():
                        p.requires_grad_(self.cfg.net.finetune_encoder)
                
                critic1 = with_encoder(critic1, encoder=shared_enc)
                critic2 = with_encoder(critic2, encoder=shared_enc)
        self.sac_trainer = SacTrainer(
            discount=self.cfg.train.sac_discount,
            alpha=self.cfg.algo.alpha,
            device=self.device,
            scale_reward=self.cfg.train.sac_scale_reward,
            target_entropy=target_entropy,
            tau=self.cfg.train.sac_tau,
            critic1=critic1,
            critic2=critic2,
            actor=skill_policy,
            lr_op=self.cfg.train.lr_op,
            sac_lr_q=self.cfg.train.sac_lr_q,
            sac_lr_a=self.cfg.train.sac_lr_a,
            policy_delay=self.cfg.train.policy_delay,
            actor_start_steps=self.cfg.train.actor_start_steps,
            safe_action_distill_weight=getattr(getattr(self.cfg, "safety", None), "distill_safe_action_weight", 0.0),
        )

    def _create_sac_critic(self, input_dim, action_dim, hidden_sizes, nonlinearity):
        simba_hp = None
        if self.cfg.net.ac_backbone == 'simba':
            simba_hp = SimBaActorCriticHP(
                actor_hidden_dim=self.cfg.net.simba_actor_hidden_dim,
                actor_num_blocks=self.cfg.net.simba_actor_num_blocks,
                critic_hidden_dim=self.cfg.net.simba_critic_hidden_dim,
                critic_num_blocks=self.cfg.net.simba_critic_num_blocks,
                mlp_ratio=self.cfg.net.simba_mlp_ratio,
                rsnorm_momentum=self.cfg.net.simba_rsnorm_momentum,
                rsnorm_eps=self.cfg.net.simba_rsnorm_eps,
                ln_eps=self.cfg.net.simba_ln_eps,
            )
        return ActorCriticModuleFactory.create_critic_core(
            backbone=self.cfg.net.ac_backbone,
            obs_dim=input_dim,
            action_dim=action_dim,
            mlp_hidden_sizes=hidden_sizes,
            mlp_nonlinearity=nonlinearity or torch.relu,
            simba_hp=simba_hp,
        )

    def _create_sac_actor(self, with_encoder, input_dim, action_dim, hidden_sizes, nonlinearity):
        simba_hp = None
        if self.cfg.net.ac_backbone == 'simba':
            simba_hp = SimBaActorCriticHP(
                actor_hidden_dim=self.cfg.net.simba_actor_hidden_dim,
                actor_num_blocks=self.cfg.net.simba_actor_num_blocks,
                critic_hidden_dim=self.cfg.net.simba_critic_hidden_dim,
                critic_num_blocks=self.cfg.net.simba_critic_num_blocks,
                mlp_ratio=self.cfg.net.simba_mlp_ratio,
                rsnorm_momentum=self.cfg.net.simba_rsnorm_momentum,
                rsnorm_eps=self.cfg.net.simba_rsnorm_eps,
                ln_eps=self.cfg.net.simba_ln_eps,
            )
        
        if self.cfg.cascade.use_cascade:
             # Use a picklable factory class instead of a closure
             stage_factory = CascadeStageFactory(
                obs_dim=self.module_obs_dim,
                skill_dim=self.cfg.algo.dim_skill,
                skill_input_dim=input_dim - self.module_obs_dim,
                num_skill_levels=self.cfg.algo.num_skill_levels,
                use_hierarchical_policy=self.cfg.algo.use_hierarchical_policy,
                use_hierarchical_skill=self.cfg.algo.use_hierarchical_skill,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                nonlinearity=nonlinearity,
                actor_init_std=self.cfg.net.actor_init_std,
                actor_max_log_std=self.cfg.net.actor_max_log_std,
                simba_hp=simba_hp,
                ac_backbone=self.cfg.net.ac_backbone,
             )

             policy_core = CascadeActor(
                stage_factory_fn=stage_factory,
                device=self.device,
                gate_type=self.cfg.cascade.cascade_gate_type,
                min_lambda=self.cfg.cascade.cascade_min_lambda,
                max_lambda=self.cfg.cascade.cascade_max_lambda,
                use_hierarchical_policy=self.cfg.algo.use_hierarchical_policy,
                use_hierarchical_skill=self.cfg.algo.use_hierarchical_skill,
                num_skill_levels=self.cfg.algo.num_skill_levels,
                dim_skill=self.cfg.algo.dim_skill,
                obs_dim=self.module_obs_dim,
                skill_input_dim=input_dim - self.module_obs_dim,
             )
        else:
            policy_core = ActorCriticModuleFactory.create_actor_core(
                backbone=self.cfg.net.ac_backbone,
                input_dim=input_dim,
                action_dim=action_dim,
                mlp_hidden_sizes=hidden_sizes,
                mlp_nonlinearity=nonlinearity,
                actor_init_std=self.cfg.net.actor_init_std,
                actor_max_log_std=self.cfg.net.actor_max_log_std,
                simba_hp=simba_hp,
            )

        # CRITICAL FIX: The user asked if Actor and Critic share the visual module (encoder).
        # In the original code: module = with_encoder(policy_core) if self.cfg.net.encoder else policy_core
        # 'with_encoder' uses 'self.shared_encoder'.
        # So yes, they share the encoder.
        # My Cascade implementation: I return 'policy_core' (CascadeActor) and then wrap it with 'with_encoder'.
        # So CascadeActor.forward receives embedding.
        # This IS correct and consistent with "Actor and Critic share visual module".
        
        module = with_encoder(policy_core) if self.cfg.net.encoder else policy_core
        
        return PolicyEx(
            name='skill_policy',
            env_spec=self._env.spec,
            module=module,
            skill_info={
                'dim_skill': self.cfg.algo.dim_skill,
                'num_skill_levels': self.cfg.algo.num_skill_levels,
                'use_hierarchical_skill': self.cfg.algo.use_hierarchical_skill,
            },
        ).to(self.device)

    def _load_actor_from_pretraining(self):
        path = self.cfg.train.skill_policy_path
        if not path:
             raise ValueError("skill_policy_path required for finetune")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

        pre_metra = load_pretrained_metra(
            os.path.dirname(path),
            device=self.device,
            skill_policy_name=os.path.basename(path),
            load_traj_encoder=False,
            freeze=False,
            eval_mode=False,
        )
        if pre_metra.dim_skill is not None and int(pre_metra.dim_skill) != int(self.cfg.algo.dim_skill):
            raise ValueError(
                f"finetune dim_skill mismatch: checkpoint={pre_metra.dim_skill}, config={self.cfg.algo.dim_skill}"
            )
        if pre_metra.discrete is not None and bool(pre_metra.discrete) != bool(self.cfg.algo.discrete):
            raise ValueError(
                f"finetune discrete mismatch: checkpoint={pre_metra.discrete}, config={self.cfg.algo.discrete}"
            )
        policy_module = getattr(pre_metra.skill_policy, "_module", pre_metra.skill_policy)
        loaded_uses_encoder = isinstance(policy_module, WithEncoder) or (
            hasattr(policy_module, "encoder")
            and hasattr(policy_module, "module")
            and callable(getattr(policy_module, "get_rep", None))
        )
        if bool(self.cfg.net.encoder) != bool(loaded_uses_encoder):
            raise ValueError(
                "[finetune] encoder mismatch between checkpointed policy and current config: "
                f"checkpoint_uses_encoder={loaded_uses_encoder}, config.encoder={bool(self.cfg.net.encoder)}"
            )
        return pre_metra.skill_policy

    def _init_kme_module(self):
        kme_components = {
            'init_kme': False,
            'idk_step_counter': 0,
            'kernel': None,
            'kme_vector': None
        }
        
        if should_use_kme(self.cfg):
            # Determine input dim for kernel
            if self.cfg.algo.idk_from == 'traj':
                input_dim = self.cfg.algo.dim_skill
            else:
                # If using raw state or shared encoder features
                input_dim = self.module_obs_dim
                
            kme_components['kernel'] = SoftIsolationKernel(
                input_dim=input_dim,
                ensemble_size=100, subsample_size=self.cfg.algo.idk_subsample_size, temperature=0.0001,
                device=self.device,
            ).to(self.device)
            
            # SoftIsolationKernel forward returns (B, ensemble_size * subsample_size)
            kme_vector_dim = kme_components['kernel'].ensemble_size * kme_components['kernel'].subsample_size
            kme_components['kme_vector'] = torch.zeros(kme_vector_dim).to(self.device)
            
        return kme_components
