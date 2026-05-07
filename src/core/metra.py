import copy
import functools
import logging
import os
import time
import types
from collections import defaultdict
from math import inf, sqrt
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import utils
from core.isolation_kernel import SoftIsolationKernel
from legacy.load_pretrain_metra import load_pretrained_metra
from core.networks import PolicyEx, ContinuousMLPQFunctionEx, GaussianMLPIndependentStdModuleEx, \
    GaussianMLPTwoHeadedModuleEx, WithEncoder, EncoderFactory
from data_structs.trajectory_batch import TrajectoryBatch
from workers.rollout import SkillRolloutWorker
from core.sac_trainer import SacTrainer
from core.skill_selector import DiscreteSkillSelector, CEMSkillSelector
from core.stage_contract import should_update_target_traj_encoder, uses_external_reward
from utils.utils import _finalize_lr, OptimizerGroupWrapper
from core.actor_critic_factory import ActorCriticModuleFactory, SimBaActorCriticHP
from core.metra_viz import plot_trajectories as _plot_trajectories
from envs.kitchen.metrics import calc_kitchen_eval_metrics
from iod.coverage_tracker import CoverageTracker

class _StopGradEncoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        # IMPORTANT: do NOT register the shared encoder as a submodule,
        # otherwise traj_encoder.state_dict()/parameters() will include it.
        # We bypass nn.Module.__setattr__ to avoid submodule registration.
        self.__dict__['_enc'] = enc

    @property
    def enc(self):
        return self.__dict__['_enc']

    def forward(self, x):
        prev_training = self.enc.training
        try:
            # Avoid BN/Dropout state updates from the traj encoder path.
            self.enc.eval()
            with torch.no_grad():
                y = self.enc(x)
        finally:
            self.enc.train(prev_training)
        return y.detach()

class DictBatchDataset:
    """Use when the input is the dict type."""
    def __init__(self, inputs, batch_size):
        self._inputs = inputs
        self._batch_size = batch_size
        self._size = list(self._inputs.values())[0].shape[0]
        if batch_size is not None:
            self._ids = np.arange(self._size)
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._size * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield self._inputs
        else:
            if update:
                self.update()
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = {
                    k: v[batch_ids]
                    for k, v in self._inputs.items()
                }
                yield batch

    def update(self):
        np.random.shuffle(self._ids)

class MeasureAndAccTime:
    def __init__(self, target):
        assert isinstance(target, list)
        assert len(target) == 1
        self._target = target

    def __enter__(self):
        self._time_enter = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target[0] += (time.time() - self._time_enter)

def compute_total_norm(parameters, norm_type=2):
    # Code adopted from clip_grad_norm_().
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def _dist_or_tensor_mean(value):
    if torch.is_tensor(value):
        return value
    return value.mean


class DRQ_METRAAgent:
    """
    DrQ agent with METRA skill discovery integration.
    Manages the actor, critic, and their updates.
    Configuration is now passed via explicit arguments (formerly Hydra).
    """
    def __init__(self,
                 env,
                 tau,
                 scale_reward,
                 target_coef,
                 replay_buffer,
                 min_buffer_size,
                 inner,
                 num_alt_samples,
                 split_group,
                 dual_lam,
                 dual_reg,
                 dual_slack,
                 dual_dist,
                 pixel_shape,
                 env_name,
                 algo,
                 env_spec,
                 skill_dynamics,
                 dist_predictor,
                 alpha,
                 time_limit,
                 n_epochs_per_eval,
                 n_epochs_per_log,
                 n_epochs_per_tb,
                 n_epochs_per_save,
                 n_epochs_per_pt_save,
                 n_epochs_per_pkl_update,
                 dim_skill,
                 num_random_trajectories,
                 num_video_repeats,
                 eval_record_video,
                 video_skip_frames,
                 eval_plot_axis,
                 name='IOD',
                 device=torch.device('cuda'),
                 sample_cpu=True,
                 num_train_per_epoch=100000,
                 discount=0.99,
                 sd_batch_norm=False,
                 skill_dynamics_obs_dim=None,
                 trans_minibatch_size=None,
                 trans_optimization_epochs=None,
                 discrete=False,
                 unit_length=False,
                 batch_size=32,
                 snapshot_dir=None,
                 use_encoder=True,
                 encoder_type='original',
                 finetune_encoder=False,
                 spectral_normalization=False,
                 model_master_nonlinearity=None,
                 model_master_dim=1024,
                 model_master_num_layers=2,
                 lr_op=None,
                 lr_te=None,
                 dual_lr=None,
                 sac_lr_q=None,
                 sac_lr_a=None,
                 seed=0,
                 grad_clip_norm=None,
                 actor_init_std=1.0,
                 actor_max_log_std=2.0,
                 use_target_traj_encoder=False,
                 use_kme=False,
                 update_idk=1000,
                 idk_subsample_size=256,
                 idk_init='gaussian',  # {'gaussian','uniform','replay'}
                 idk_from='traj',
                 idk_groups = 1,
                 kernel_map=False,
                 use_novelty_reward = False,
                 stage = 'pre_training',
                 skill_policy_path = '',
                 policy_delay=1,
                 actor_start_steps=0,
                 actor_critic_backbone='mlp',     # choices: ['mlp', 'simba']
                 simba_actor_hidden_dim=128,
                 simba_actor_num_blocks=1,
                 simba_critic_hidden_dim=512,
                 simba_critic_num_blocks=2,
                 simba_mlp_ratio=4,
                 simba_rsnorm_momentum=0.999,
                 simba_rsnorm_eps=1e-5,
                 simba_ln_eps=1e-5,
                 ):
        self._env = env
        self.env_name = env_name
        self.coverage_tracker = CoverageTracker(env_name)
        self.algo = algo
        self.seed = seed
        self.enable_logging = True

        self.step_itr = 0
        self.total_train_steps = 0 # Track total training steps (gradient updates)
        self.policy_delay = policy_delay
        self.actor_start_steps = actor_start_steps
        self.snapshot_dir = snapshot_dir

        self.discount = discount
        self.time_limit = time_limit

        self.device = device
        self.sample_cpu = sample_cpu
        self.stage = stage
        if self.stage == 'finetune':
            self.skill_policy_path = skill_policy_path
        # skill_policy
        if model_master_nonlinearity == 'relu':
            nonlinearity = torch.relu
        elif model_master_nonlinearity == 'tanh':
            nonlinearity = torch.tanh
        else:
            nonlinearity = None
        self.use_encoder = use_encoder
        self.encoder_type = encoder_type
        self.finetune_encoder = finetune_encoder

        example_ob = env.reset()
        if self.use_encoder: # for pixels input
            self.shared_encoder = EncoderFactory.create(
                encoder_type=self.encoder_type,
                pixel_shape=pixel_shape,
                finetune=self.finetune_encoder,
                spectral_normalization=spectral_normalization,
                device=self.device
            )
            self.shared_encoder.to(self.device)

            def make_encoder(**kwargs):
                return self.shared_encoder

            def with_encoder(module, encoder=None):
                if encoder is None:
                    encoder = self.shared_encoder

                return WithEncoder(encoder=encoder, module=module)

            example_encoder = self.shared_encoder
            module_obs_dim = example_encoder(torch.as_tensor(example_ob["image"]).to(self.device).float().unsqueeze(0)).shape[-1]
        else:
            # 1. 从 reset 的 obs 里抽出 state 向量
            state = self._extract_state_from_obs(example_ob["info"]['state'])   # 下面会给实现
            state = np.asarray(state, dtype=np.float32)
            module_obs_dim = int(np.prod(state.shape))

            # 2. 保持接口兼容：with_encoder 变成 identity
            make_encoder = None
            with_encoder = None

        policy_q_input_dim = module_obs_dim + dim_skill
        action_dim = self._env.spec.action_space.flat_dim
        master_dims = [model_master_dim] * model_master_num_layers
        self.optimizers_dict = {}
        self.param_modules = {}
        # traj_encoder
        self.use_target_traj_encoder = use_target_traj_encoder
        if self.stage == "pre_training":
            self.spectral_normalization = spectral_normalization
            self._init_traj_encoder(
                make_encoder, with_encoder, module_obs_dim, dim_skill, master_dims, nonlinearity, dual_lam,
                lr_te, dual_lr
            )
        if skill_dynamics is not None:
            self.skill_dynamics = skill_dynamics.to(self.device)
            self.param_modules['skill_dynamics'] = self.skill_dynamics
        if dist_predictor is not None:
            self.dist_predictor = dist_predictor.to(self.device)
            self.param_modules['dist_predictor'] = self.dist_predictor

        if skill_dynamics is not None:
            self.optimizers_dict.update({
                'skill_dynamics': torch.optim.Adam([
                    {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(lr_te)},
                ]),
            })
        if dist_predictor is not None:
            self.optimizers_dict.update({
                'dist_predictor': torch.optim.Adam([
                    {'params': dist_predictor.parameters(), 'lr': _finalize_lr(lr_op)},
                ]),
            })

        self.alpha = alpha
        self.name = name
        self.grad_clip_norm = grad_clip_norm

        self.dim_skill = dim_skill
        self.actor_init_std = actor_init_std
        self.actor_critic_backbone = str(actor_critic_backbone)
        if self.actor_critic_backbone not in ('mlp', 'simba'):
            raise ValueError(f"actor_critic_backbone must be in ['mlp','simba'], got {self.actor_critic_backbone!r}.")

        self.simba_hp = None
        if self.actor_critic_backbone == 'simba':
            # Paper defaults: actor 128x1, critic 512x2; keep overridable for VRAM constraints.
            self.simba_hp = SimBaActorCriticHP(
                actor_hidden_dim=int(simba_actor_hidden_dim),
                actor_num_blocks=int(simba_actor_num_blocks),
                critic_hidden_dim=int(simba_critic_hidden_dim),
                critic_num_blocks=int(simba_critic_num_blocks),
                mlp_ratio=int(simba_mlp_ratio),
                rsnorm_momentum=float(simba_rsnorm_momentum),
                rsnorm_eps=float(simba_rsnorm_eps),
                ln_eps=float(simba_ln_eps),
            )


        self.actor_max_log_std = actor_max_log_std

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_log = n_epochs_per_log
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.n_epochs_per_pkl_update = n_epochs_per_pkl_update
        self.num_random_trajectories = num_random_trajectories
        self.num_video_repeats = num_video_repeats
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames
        self.eval_plot_axis = eval_plot_axis

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self.discrete = discrete
        self.unit_length = unit_length
        self.batch_size = batch_size


        self._optimizer = OptimizerGroupWrapper(
            optimizers=self.optimizers_dict,
            max_optimization_epochs=None,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        target_entropy = -np.prod(self._env_spec.action_space.shape).item()  * target_coef

        self._init_sac(alpha, device, scale_reward,lr_op, sac_lr_q, sac_lr_a, target_entropy, tau,
                       env_spec, with_encoder, policy_q_input_dim, action_dim, master_dims, nonlinearity, dim_skill,
                       policy_delay, actor_start_steps
                       )

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None

        self._start_time = time.time()
        self.total_env_steps = 0
        self.total_epoch = 0

        self.rollout_worker = SkillRolloutWorker(self.seed, time_limit=self.time_limit, cur_extra_keys=['skill'], pixeled=use_encoder)

        self.writer = None
        self.logger = None
        self.use_novelty_reward = use_novelty_reward
        # set kernel mean embedding
        self.use_kme = use_kme
        self.kernel_map_obs = kernel_map
        if self.use_kme:
            self.init_kme = False
            self.update_idk = int(update_idk)
            self.idk_subsample_size = int(idk_subsample_size)
            self.idk_init = idk_init
            self.idk_from = idk_from
            self.idk_step_counter = 0
            self.beta = 0.9
            self.idk_groups = idk_groups
            self.path_datas = []
            if not hasattr(self, 'kernel'):
                self.kernel = SoftIsolationKernel(
                    input_dim=self.dim_skill if idk_from=='traj' else self.encoder.feature_dim,
                    ensemble_size=100, subsample_size=self.idk_subsample_size, temperature=0.0001,
                    device=self.device,
                ).to(self.device)
            self.kme_vector = torch.zeros(self.idk_subsample_size * self.kernel.ensemble_size).to(self.device)

    def find_best_skill(self):
        # zero_training 或 dim_skill=0 时，不存在 skill 搜索问题
        if self.stage == 'zero_training' or int(getattr(self, 'dim_skill', 0)) == 0:
            return np.zeros((0,), dtype=np.float32)
        if self.discrete:
            selector = DiscreteSkillSelector(
                env=self._env,
                actor=self.sac_trainer.skill_policy,
                worker=self.rollout_worker,
                device=self.device,
                dim_skill=self.dim_skill,
                logger=self.logger
            )
        else:
            selector = CEMSkillSelector(
                env=self._env,
                actor=self.sac_trainer.skill_policy,
                worker=self.rollout_worker,
                device=self.device,
                dim_skill=self.dim_skill,
                logger=self.logger
            )
        return selector.select()

    def _init_traj_encoder(self, make_encoder, with_encoder, module_obs_dim, dim_skill, master_dims, nonlinearity,
                           dual_lam, lr_te, dual_lr):
        dim_skill = int(dim_skill)
        if dim_skill <= 0:
            raise ValueError(f"[METRA] pre_training requires dim_skill > 0, got dim_skill={dim_skill}.")

        # --- traj head ---
        traj_head = GaussianMLPIndependentStdModuleEx(
            input_dim=module_obs_dim,
            output_dim=dim_skill,
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
            spectral_normalization=getattr(self, "spectral_normalization", False),
        ).to(self.device)

        # --- stop-grad shared encoder wrapper (IMPORTANT: do NOT register encoder as a submodule) ---
        if self.use_encoder:
            base_encoder = make_encoder()  # in your metra.py, make_encoder() returns self.shared_encoder
            stop_grad_encoder = _StopGradEncoder(base_encoder).to(self.device)
            self.traj_encoder = WithEncoder(encoder=stop_grad_encoder, module=traj_head).to(self.device)
        else:
            self.traj_encoder = traj_head
        self.traj_encoder.eval()

        # --- target traj encoder: copy head only ---
        self.target_traj_encoder = None
        if getattr(self, "use_target_traj_encoder", False):
            if isinstance(self.traj_encoder, WithEncoder):
                self.target_traj_encoder = WithEncoder(
                    encoder=self.traj_encoder.encoder,                # same stop-grad wrapper
                    module=copy.deepcopy(self.traj_encoder.module),   # head copy only
                ).to(self.device)
                for p in self.target_traj_encoder.module.parameters():
                    p.requires_grad_(False)
            else:
                self.target_traj_encoder = copy.deepcopy(self.traj_encoder).to(self.device)
                for p in self.target_traj_encoder.parameters():
                    p.requires_grad_(False)
            self.target_traj_encoder.eval()

        # --- dual lambda ---
        self.dual_lam = utils.ParameterModule(torch.tensor([np.log(float(dual_lam))], dtype=torch.float32)).to(self.device)

        self.param_modules.update({
            'traj_encoder': self.traj_encoder,
            'dual_lam': self.dual_lam,
        })

        # --- optimizer: head only ---
        if isinstance(self.traj_encoder, WithEncoder):
            te_params = list(self.traj_encoder.module.parameters())
        else:
            te_params = list(self.traj_encoder.parameters())

        self.optimizers_dict.update({
            'traj_encoder': torch.optim.Adam([{'params': te_params, 'lr': _finalize_lr(lr_te)}]),
            'dual_lam': torch.optim.Adam([{'params': self.dual_lam.parameters(), 'lr': _finalize_lr(dual_lr)}]),
        })

    def _init_sac(self, alpha, device, scale_reward, lr_op, sac_lr_q, sac_lr_a, target_entropy, tau,
                  env_spec, with_encoder, policy_q_input_dim, action_dim, master_dims, nonlinearity, dim_skill,
                  policy_delay, actor_start_steps):
        """
        stage:
          - pre_training: skill-conditioned SAC/DrQ-SAC
          - finetune: load skill_policy from pre_training, share encoder with critics
          - zero_training: NO skill, dim_skill must be 0, randomly init policy, train plain SAC/DrQ-SAC
        """

        dim_skill = int(dim_skill)

        if self.stage == 'zero_training' and dim_skill != 0:
            raise ValueError(f"[zero_training] requires dim_skill=0, got dim_skill={dim_skill}.")

        if self.stage == 'pre_training' and dim_skill <= 0:
            raise ValueError(f"[pre_training] requires dim_skill>0, got dim_skill={dim_skill}.")

        if self.stage == 'finetune' and dim_skill <= 0:
            raise ValueError(f"[finetune] expects dim_skill>0. For dim_skill=0 use stage='zero_training'.")

        critic1 = self._create_sac_critic(policy_q_input_dim, action_dim, master_dims, nonlinearity)
        critic2 = self._create_sac_critic(policy_q_input_dim, action_dim, master_dims, nonlinearity)

        # Ensure shared encoder finetune state (pre_training/zero_training)
        if self.use_encoder and self.stage in ('pre_training', 'zero_training'):
            if hasattr(self.shared_encoder, "set_finetune"):
                self.shared_encoder.set_finetune(bool(getattr(self, "finetune_encoder", False)))
            else:
                req = bool(getattr(self, "finetune_encoder", False))
                for p in self.shared_encoder.parameters():
                    p.requires_grad_(req)

        if self.stage in ('pre_training', 'zero_training'):
            # randomly init actor
            skill_policy = self._create_sac_actor(
                env_spec, with_encoder, policy_q_input_dim, action_dim, master_dims, nonlinearity, dim_skill
            )
            if self.use_encoder:
                critic1 = with_encoder(critic1)  # default uses self.shared_encoder
                critic2 = with_encoder(critic2)

        elif self.stage == 'finetune':
            skill_policy = self._load_actor_from_pretraining().to(self.device)
            skill_policy.train()

            if self.use_encoder:
                if not hasattr(skill_policy, "_module") or not hasattr(skill_policy._module, "encoder"):
                    raise AttributeError("[finetune] loaded skill_policy does not expose _module.encoder.")

                shared_enc = skill_policy._module.encoder  # SHARE, do NOT deepcopy

                # align finetune/freeze
                if hasattr(shared_enc, "set_finetune"):
                    shared_enc.set_finetune(bool(getattr(self, "finetune_encoder", False)))
                else:
                    req = bool(getattr(self, "finetune_encoder", False))
                    for p in shared_enc.parameters():
                        p.requires_grad_(req)

                critic1 = with_encoder(critic1, encoder=shared_enc)
                critic2 = with_encoder(critic2, encoder=shared_enc)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        self.sac_trainer = SacTrainer(
            discount=self.discount,
            alpha=alpha,
            device=device,
            scale_reward=scale_reward,
            target_entropy=target_entropy,
            tau=tau,
            critic1=critic1,
            critic2=critic2,
            actor=skill_policy,
            lr_op=lr_op,
            sac_lr_q=sac_lr_q,
            sac_lr_a=sac_lr_a,
            policy_delay = policy_delay,
            actor_start_steps = actor_start_steps,
        )

    def _load_actor_from_pretraining(self, skill_policy_path=None):
        """
        Load pretrained METRA and return its skill_policy.

        Patched:
          - allow passing skill_policy_path explicitly
          - validate path existence early for clearer errors
        """
        if skill_policy_path is None or skill_policy_path == "":
            skill_policy_path = getattr(self, "skill_policy_path", "")

        if not skill_policy_path:
            raise ValueError("skill_policy_path is empty in finetune stage; cannot load pretrained skill_policy.")

        if not os.path.exists(skill_policy_path):
            raise FileNotFoundError(f"skill_policy_path not found: {skill_policy_path}")

        pre_metra = load_pretrained_metra(
            os.path.dirname(skill_policy_path),
            device=self.device,
            skill_policy_name=os.path.basename(skill_policy_path),
            load_traj_encoder=False,
            freeze=False,
            eval_mode=False,
        )

        skill_policy = pre_metra.skill_policy

        # Sync dim_skill/discrete if present in pretrained config
        if getattr(pre_metra, "dim_skill", None):
            self.dim_skill = pre_metra.dim_skill
            self.discrete = pre_metra.discrete

        return skill_policy


    def _create_sac_critic(self, policy_q_input_dim, action_dim, master_dims, nonlinearity):
        """Create critic core (Q-function). Backbone is selected via `self.actor_critic_backbone`."""
        return ActorCriticModuleFactory.create_critic_core(
            backbone=self.actor_critic_backbone,
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            mlp_hidden_sizes=master_dims,
            mlp_nonlinearity=nonlinearity or torch.relu,
            simba_hp=self.simba_hp,
        )

    def _create_sac_actor(self, env_spec, with_encoder, policy_q_input_dim, action_dim, master_dims, nonlinearity, dim_skill):
        """Create actor (policy). Backbone is selected via `self.actor_critic_backbone`."""
        policy_core = ActorCriticModuleFactory.create_actor_core(
            backbone=self.actor_critic_backbone,
            input_dim=policy_q_input_dim,
            action_dim=action_dim,
            mlp_hidden_sizes=master_dims,
            mlp_nonlinearity=nonlinearity,
            actor_init_std=self.actor_init_std,
            actor_max_log_std=self.actor_max_log_std,
            simba_hp=self.simba_hp,
        )

        # IMPORTANT: keep METRA's existing input pipeline:
        # SimBa/MLP consumes the same (encoded_image + state (+ skill)) vector as before.
        module = with_encoder(policy_core) if self.use_encoder else policy_core

        return PolicyEx(
            name='skill_policy',
            env_spec=env_spec,
            module=module,
            skill_info={'dim_skill': dim_skill},
        ).to(self.device)

    def _extract_state_from_obs(self, obs):

        # 情况 1: obs 是 dict
        if isinstance(obs, dict):
            # 1) 先试几个常见 key
            for key in ("state", "obs", "observation"):
                if key in obs:
                    return np.asarray(obs[key], dtype=np.float32)

            # 2) 兜底：找第一个非 image 的 1D 向量
            candidate = None
            for k, v in obs.items():
                if k == "image":
                    continue
                arr = np.asarray(v)
                if arr.ndim == 1:
                    candidate = arr.astype(np.float32)
                    break

            if candidate is None:
                raise RuntimeError(
                    f"use_encoder=False 但在 obs={list(obs.keys())} 里找不到合适的一维状态向量，"
                    f"请检查环境返回或手动修改 _extract_state_from_obs 的逻辑。"
                )

            return candidate

        # 情况 2: obs 不是 dict，直接当作 state
        return np.asarray(obs, dtype=np.float32)


    @torch.no_grad()
    def update_target_traj_encoder(self):
        """Scheme A: 只软更新 traj head，不更新 encoder。"""
        if not getattr(self, "use_target_traj_encoder", False):
            return
        if getattr(self, "target_traj_encoder", None) is None:
            return

        tau = self.tau

        if isinstance(self.traj_encoder, WithEncoder) and isinstance(self.target_traj_encoder, WithEncoder):
            src_params = list(self.traj_encoder.module.parameters())
            tgt_params = list(self.target_traj_encoder.module.parameters())
        else:
            src_params = list(self.traj_encoder.parameters())
            tgt_params = list(self.target_traj_encoder.parameters())

        for p, tp in zip(src_params, tgt_params):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)


    @torch.no_grad()
    def _ik_input_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, C, H, W) or already-processed features, depending on your pipeline.
        返回 φ(s)。与 _update_dot_rewards 中计算 cur_z/next_z 的路径一致。
        """
        traj_encoder = self.target_traj_encoder if self.use_target_traj_encoder  else self.traj_encoder
        traj_encoder.eval()
        if self.idk_from == 'traj':
            feat = obs                                # (B, F)
            dist = traj_encoder(feat)            # Normal(...)
            phi = dist.mean                           # (B, Z)
        else:
            phi = obs                                 # (B, F) 直接用 encoder 特征
        return phi

    @torch.no_grad()
    def _build_idk_initial(self):
        """
        根据 idk_init 生成一组 anchors（φ(s)），并喂给 kernel.fit(...)
        """
        D = self.dim_skill
        M = self.idk_subsample_size
        # 1) 用随机分布初始化
        if self.idk_init in ('gaussian','uniform'):
            if self.idk_init == 'gaussian':
                anchors = torch.randn(M, D, device=self.device)
                data = torch.randn(self.replay_buffer._capacity // 10, D, device=self.device)
            else:
                anchors = torch.empty(M, D, device=self.device).uniform_(-1.0, 1.0)
                data = torch.empty(self.replay_buffer._capacity // 10, D, device=self.device)
            self.kernel.fit(anchors)

            self.kme_vector = self.kernel.kernel_mean(data, groups=self.idk_groups) / sqrt(self.kernel.ensemble_size)
            return

        # 2) 从 replay 里抽样
        anchors = self._sample_ik_state_from_replay(M)
        if anchors is None:
            anchors = torch.randn(M, D, device=self.device)

        self.kernel.fit(anchors)
        self.rebuild_idk_mean()

    @torch.no_grad()
    def _maybe_refresh_idk_from_replay(self, metrics: dict=None):
        """
        每过 update_idk 个 TE 更新，就从 replay 中抽样状态 -> φ(s)，重建 IDK 基字典。
        """
        if self.update_idk <= 0:
            return
        self.idk_step_counter += 1
        if (self.idk_step_counter % self.update_idk) != 0:
            return
        anchors = self._sample_ik_state_from_replay(self.idk_subsample_size)
        if anchors is None:
            # 没有足够数据，不刷新
            if metrics is not None:
                metrics['idk_refresh_skipped'] = 1
            return

        # 是否对 anchors 做零均值/单位方差化
        # anchors = (anchors - anchors.mean(dim=0, keepdim=True)) / (anchors.std(dim=0, keepdim=True) + 1e-6)
        self.kernel.fit(anchors)
        # if self.beta < 1.0:
        #     self.rebuild_idk_mean()
        print(" UP date IDK\n\n\n\n")

    @torch.no_grad()
    def _sample_ik_state_from_replay(self, n: int):
        """
        随机从 replay 取 n 条 obs，转成 torch 张量，编码为 φ(s)。
        如果 buffer 样本不够，返回 None。
        """
        if (not hasattr(self, 'replay_buffer')) or (self.replay_buffer is None):
            raise "No replay buffer"
        if len(self.replay_buffer) < max(n, 256):  # 至少有一定量再抽
            raise "Too few samples"

        batch = self.replay_buffer.sample_transitions(n)
        buffer_obs = batch.get('obs', None)
        if buffer_obs.dtype == np.uint8:
            obs = torch.from_numpy(buffer_obs).float().to(self.device)  # 若是像素，做归一化
        elif buffer_obs.dtype in [np.float64, np.float32] :
            obs = torch.from_numpy(buffer_obs).float().to(self.device)

        laten_state = self._ik_input_from_obs(obs)                # (n, D)
        return laten_state

    @torch.no_grad()
    def rebuild_idk_mean(self):
        if (not hasattr(self, 'replay_buffer')) or (self.replay_buffer is None):
            raise "No replay buffer"

        kernel_embd = []
        for _ in range(200):
            N = len(self.replay_buffer) // 200 # 是否需要取全部的replay buffer 做平均
            batch = self.replay_buffer.sample_transitions(N)
            all_obs = batch['obs']
            if isinstance(all_obs, np.ndarray):
                all_obs = torch.from_numpy(all_obs).float().to(self.device)  # 若是像素，做归一化
            else:
                all_obs = all_obs.to(self.device)
            all_embd = self._ik_input_from_obs(all_obs)
            kernel_embd.append(self.kernel.kernel_mean(all_embd, groups=self.idk_groups))
        self.kme_vector = torch.mean(torch.stack(kernel_embd), dim = 0) / sqrt(self.kernel.ensemble_size)

    @property
    def policy(self):
        return {
            'skill_policy': self.sac_trainer.skill_policy,
        }

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p


    #########################################################################################################
    #                                                                                                       #
    #                                        interacting with env                                           #
    #                                                                                                       #
    #########################################################################################################
    def _generate_skill_extras(self, skills):
        return [{'skill': skill} for skill in skills]

    def _get_train_trajectories_kwargs(self):
        # zero_training / dim_skill=0：不传 skill，不让 worker 记录 skill
        if self.stage == 'zero_training' or int(getattr(self, "dim_skill", 0)) == 0:
            if hasattr(self, "rollout_worker") and hasattr(self.rollout_worker, "_cur_extra_keys"):
                self.rollout_worker._cur_extra_keys = []
            return {}

        # finetune：固定 best_skill
        if self.stage == 'finetune':
            if not hasattr(self, "best_skill"):
                self.best_skill = self.find_best_skill()
            extras = self._generate_skill_extras(
                np.repeat(self.best_skill[None, :], self.batch_size, axis=0)
            )
            return dict(extras=extras)

        # pre_training：随机 skills
        if hasattr(self, "rollout_worker") and hasattr(self.rollout_worker, "_cur_extra_keys"):
            self.rollout_worker._cur_extra_keys = ['skill']

        if getattr(self, "discrete", False):
            extras = self._generate_skill_extras(
                np.eye(self.dim_skill)[np.random.randint(0, self.dim_skill, self.batch_size)]
            )
        else:
            random_skills = np.random.randn(self.batch_size, self.dim_skill).astype(np.float32)
            if getattr(self, "unit_length", False):
                n = np.linalg.norm(random_skills, axis=1, keepdims=True) + 1e-8
                random_skills = random_skills / n
            extras = self._generate_skill_extras(random_skills)

        return dict(extras=extras)

    def _get_train_trajectories(self):
        default_kwargs = dict(
            batch_size=self.batch_size,
            deterministic_policy=False,
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs())

        paths = self._get_trajectories(**kwargs)

        return paths

    def _get_trajectories(self,
                          batch_size=None,
                          deterministic_policy=False,
                          extras=None, state_record_pixeled=False) -> List[dict]:
        if batch_size is None:
            batch_size = len(extras)
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories = self.obtain_exact_trajectories(
                env=self._env,
                policy=self.sac_trainer.skill_policy,
                batch_size=batch_size,
                extras=extras,
                deterministic_policy=deterministic_policy,
                state_record_pixeled=state_record_pixeled,
            )
        print(f'_get_trajectories {time_get_trajectories[0]}s')

        # for traj in trajectories:
        #     for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
        #         if key not in traj['env_infos']:
        #             continue

        return trajectories

    def obtain_exact_trajectories(self, env, policy, batch_size, extras, deterministic_policy=False, state_record_pixeled = False):
        batches = []
        for i in range(batch_size):
            extra = None if self.stage == 'zero_training' else extras[i]
            batch = self.rollout_worker.rollout(env, policy, extra, deterministic_policy=deterministic_policy, state_record_pixeled = state_record_pixeled)
            batches.append(batch)
        trajectories = TrajectoryBatch.concatenate(*batches)
        paths = trajectories.to_trajectory_list()
        return paths

    #########################################################################################################
    #                                                                                                       #
    #                                        processing data                                                #
    #                                                                                                       #
    #########################################################################################################
    def _get_mini_tensors(self, epoch_data):
        num_transitions = len(epoch_data['actions'])
        idxs = np.random.choice(num_transitions, self._trans_minibatch_size)

        data = {}
        for key, value in epoch_data.items():
            data[key] = value[idxs]

        return data

    def _get_concat_obs(self, obs, skill):
        """
        obs: torch.Tensor [B, D]
        skill: torch.Tensor [B, dim_skill] 或 None
        """
        if skill is None:
            if int(getattr(self, "dim_skill", 0)) == 0:
                return obs
            raise ValueError("[METRA] skill is None but dim_skill > 0.")
        return utils.get_torch_concat_obs(obs, skill)

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = param_dict[k].detach().cpu()
            else:
                param_dict[k] = param_dict[k].detach()
        return param_dict

    def process_samples(self, paths):
        data = defaultdict(list)
        for path in paths:
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(utils.discount_cumsum(path['rewards'], self.discount))
            # if 'ori_obs' in path['env_infos']:
            #     data['ori_obs'].append(path['env_infos']['ori_obs'])
            # if 'next_ori_obs' in path['env_infos']:
            #     data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])
            if 'pre_tanh_value' in path['agent_infos']:
                data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
            if 'log_prob' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_prob'])
            if 'skill' in path['agent_infos']:
                data['skills'].append(path['agent_infos']['skill'])
                data['next_skills'].append(np.concatenate([path['agent_infos']['skill'][1:], path['agent_infos']['skill'][-1:]], axis=0))

        return data

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                if self.use_kme and self.init_kme:
                    traj_obs = path['obs']
                    if traj_obs.dtype in [np.uint8, np.float32, np.float64]:
                        traj_obs = torch.from_numpy(traj_obs).float().to(self.device)  # 若是像素，做归一化
                    else:
                        traj_obs = traj_obs.to(self.device)
                    path['skill_kme'] = np.tile(
                        self.kernel.kernel_mean(self._ik_input_from_obs(traj_obs), groups=10).to('cpu').numpy() / sqrt(self.kernel.ensemble_size),
                        (len(path['obs']), 1)
                    )
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'skill' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    #########################################################################################################
    #                                                                                                       #
    #                                               training                                                #
    #                                                                                                       #
    #########################################################################################################
    def train(self, n_epochs):
        last_return = None
        if 'finetune' == self.stage:
            self.best_skill = self.find_best_skill()
        with utils.GlobalContext({'phase': 'train', 'policy': 'sampling'}):
            self.logger.info('Obtaining samples...')
            for epoch in tqdm(range(n_epochs)):
                self.logger.info('epoch #%d | ' % epoch)
                self._itr_start_time = time.time()
                self.total_epoch = epoch

                for p in self.policy.values():
                    p.eval()
                if 'pre_training' == self.stage:
                    self.traj_encoder.eval()

                if self.n_epochs_per_eval != 0 and ( self.step_itr + 1 ) % self.n_epochs_per_eval == 0:
                    self._evaluate_policy()

                for p in self.policy.values():
                    p.train()
                if 'pre_training' == self.stage:
                    self.traj_encoder.train()

                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        step_paths = self._get_train_trajectories()
                    self.total_env_steps += sum([step_path['dones'].shape[0] for step_path in step_paths])
                    last_return = self.train_once(
                        self.step_itr,
                        step_paths,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )

                self.step_itr += 1

                new_save = (self.n_epochs_per_save != 0 and self.step_itr % self.n_epochs_per_save == 0)
                pt_save = (self.n_epochs_per_pt_save != 0 and self.step_itr % self.n_epochs_per_pt_save == 0)
                if new_save or pt_save:
                    self.save(epoch, new_save=new_save, pt_save=pt_save)

                if self.enable_logging:
                    if self.step_itr % self.n_epochs_per_log == 0:
                        self.log_diagnostics(pause_for_plot=False)
                        if self.n_epochs_per_tb is None:
                            self.writer.flush()
                        else:
                            if self.step_itr <= 0 or (self.n_epochs_per_tb != 0 and self.step_itr % self.n_epochs_per_tb == 0):
                                self.writer.flush()
                            else:
                                print('Dump text csv std at', self.step_itr)

        return last_return

    def train_once(self, itr, paths, extra_scalar_metrics={}):
        logging_enabled = ((self.step_itr + 1) % self.n_epochs_per_log == 0)

        self.coverage_tracker.update_train_paths(paths)
        data = self.process_samples(paths)

        time_computing_metrics = [0.0]
        time_training = [0.0]
        if self.use_kme:
            self._maybe_refresh_idk_from_replay()
        with MeasureAndAccTime(time_training):
            metrics = self._train_once_inner(data)

        performance = utils.log_performance_ex(
            itr,
            batch=TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )
        discounted_returns = performance['discounted_returns']
        undiscounted_returns = performance['undiscounted_returns']
        success_rate = performance['success_rate']
        success_tasks = performance['success_tasks']
        prefix = utils.get_metric_prefix() + self.name + '/'
        self.writer.add_scalar(prefix + 'AverageExternalDiscountedReturn', np.mean(discounted_returns), self.step_itr)
        self.writer.add_scalar(prefix + 'AverageExternalReturn', np.mean(undiscounted_returns), self.step_itr)
        if success_rate is not None:
            self.writer.add_scalar(prefix + 'SuccessRate', success_rate, self.step_itr)
        if success_tasks is not None:
            self.writer.add_scalar(prefix + 'SuccessTasks', success_tasks, self.step_itr)
        if logging_enabled:
            for k in metrics.keys():
                if metrics[k].numel() == 1:
                    self.writer.add_scalar(prefix + f'{k}', metrics[k].item(), self.step_itr)
                else:
                    self.writer.add_scalar(prefix + f'{k}', metrics[k].mean(), self.step_itr)  # Use mean for arrays
            with torch.no_grad():
                total_norm = compute_total_norm(self.all_parameters())
                self.writer.add_scalar(prefix + 'TotalGradNormAll', total_norm.item(), self.step_itr)
                for key, module in self.param_modules.items():
                    total_norm = compute_total_norm(module.parameters())
                    self.writer.add_scalar(prefix + f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}', total_norm.item(), self.step_itr)
            for k, v in extra_scalar_metrics.items():
                self.writer.add_scalar(prefix + k, v, self.step_itr)
            self.writer.add_scalar(prefix + 'TimeComputingMetrics', time_computing_metrics[0], self.step_itr)
            self.writer.add_scalar(prefix + 'TimeTraining', time_training[0], self.step_itr)

            path_lengths = [
                len(path['actions'])
                for path in paths
            ]
            self.writer.add_scalar(prefix + 'PathLengthMean', np.mean(path_lengths), self.step_itr)
            self.writer.add_scalar(prefix + 'PathLengthMax', np.max(path_lengths), self.step_itr)
            self.writer.add_scalar(prefix + 'PathLengthMin', np.min(path_lengths), self.step_itr)

            self.writer.add_histogram(prefix + 'ExternalDiscountedReturns', np.asarray(discounted_returns), self.step_itr)
            self.writer.add_histogram(prefix + 'ExternalUndiscountedReturns', np.asarray(undiscounted_returns), self.step_itr)

        return np.mean(undiscounted_returns)

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            print(f"Current buffer size: {self.replay_buffer.n_transitions_stored}")
            if self.use_kme and self.kernel_map_obs:
                self.path_datas.append(path_data)
            return {}
        if self.use_kme and not self.init_kme and 'pre_training' == self.stage:
            print(f"Initializing KME mode")
            # 初次构建 IDK
            self._build_idk_initial()
            self.init_kme = True
            if self.kernel_map_obs:
                self.replay_buffer.clear()
                for  traj_data in self.path_datas:
                    self._update_replay_buffer(traj_data)
                self.path_datas = []
        epoch_data = self._flatten_data(path_data)

        metrics = self._train_components(epoch_data)

        return metrics

    def _train_components(self, epoch_data):
        for _ in range(self._trans_optimization_epochs):
            self.total_train_steps += 1
            metrics = {}

            if self.replay_buffer is None: # on policy training
                v = self._get_mini_tensors(epoch_data)
            else: # off policy training
                v = self._sample_replay_buffer()
            if 'pre_training' == self.stage:
                self._optimize_te(metrics, v)
            self._normalize_sac_scalars(v)
            if uses_external_reward(self.stage):
                self._update_external_rewards(metrics, v)
            else:
                self._update_rewards(metrics, v)
            self._optimize_op(metrics, v, self.total_train_steps)
            if should_update_target_traj_encoder(self.stage) and self.use_target_traj_encoder:
                self.update_target_traj_encoder()
        return metrics

    def _gradient_descent(self, loss, optimizer_keys, metrics=None):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._record_grad_norms(metrics, optimizer_keys)
        self._optimizer.step(keys=optimizer_keys)

    def _record_grad_norms(self, metrics, optimizer_keys):
        if metrics is None:
            return
        name_map = {
            'traj_encoder': 'TotalGradNormTrajEncoder',
            'dual_lam': 'TotalGradNormDualLam',
        }
        params = []
        for key in optimizer_keys:
            key_params = list(self._optimizer.target_parameters(keys=[key]))
            params.extend(key_params)
            metric_key = name_map.get(key)
            if metric_key is not None:
                metrics[metric_key] = compute_total_norm(key_params).detach()
        if params:
            metrics['TotalGradNormAll'] = compute_total_norm(params).detach()

    def _optimize_te(self, metrics, internal_vars):
        zero = internal_vars['obs'].detach().new_zeros(())
        metrics.setdefault('TotalGradNormAll', zero)
        metrics.setdefault('TotalGradNormTrajEncoder', zero)
        metrics.setdefault('TotalGradNormDualLam', zero)
        self._update_loss_te(metrics, internal_vars)

        self._gradient_descent(
            metrics['LossTe'],
            optimizer_keys=['traj_encoder'],
            metrics=metrics,
        )

        if self.dual_reg:
            self._update_loss_dual_lam(metrics, internal_vars)
            self._gradient_descent(
                metrics['LossDualLam'],
                optimizer_keys=['dual_lam'],
                metrics=metrics,
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    metrics['LossDp'],
                    optimizer_keys=['dist_predictor'],
                    metrics=metrics,
                )

    def _optimize_op(self, metrics, internal_vars, step):
        B = internal_vars['obs'].shape[0]
        dev = internal_vars['obs'].device

        # ensure skills exist (even empty)
        if 'skills' not in internal_vars or internal_vars['skills'] is None:
            internal_vars['skills'] = torch.zeros((B, 0), device=dev, dtype=torch.float32)
        if 'next_skills' not in internal_vars or internal_vars['next_skills'] is None:
            internal_vars['next_skills'] = torch.zeros((B, 0), device=dev, dtype=torch.float32)

        # if DrQ augment keys exist, mirror empty skills
        if 'obs_aug' in internal_vars and ('skills_aug' not in internal_vars or internal_vars['skills_aug'] is None):
            internal_vars['skills_aug'] = internal_vars['skills']
        if 'next_obs_aug' in internal_vars and ('next_skills_aug' not in internal_vars or internal_vars['next_skills_aug'] is None):
            internal_vars['next_skills_aug'] = internal_vars['next_skills']

        processed_obs = self.sac_trainer.skill_policy.process_observations(internal_vars['obs'])
        next_processed_obs = self.sac_trainer.skill_policy.process_observations(internal_vars['next_obs'])

        processed_cat_obs = self._get_concat_obs(processed_obs, internal_vars['skills'])
        next_processed_cat_obs = self._get_concat_obs(next_processed_obs, internal_vars['next_skills'])

        self.sac_trainer._optimize_once(metrics, internal_vars, processed_cat_obs, next_processed_cat_obs, step)

    def _update_distributional_novelty_rewards(self, metrics, v):
        cur_z, next_z, skills = self._get_kernel_maps(v)
        kme_map = self.kme_vector
        rewards = torch.matmul(next_z , kme_map)
        # print(f"kernel reward= {reward.mean():.10f}")
        rewards = torch.clamp(- torch.log(rewards), min = 1, max=5)
        metrics.update({
            'NoveltyRewardMean': rewards.mean(),
            'NoveltyRewardStd': rewards.std(),
        })
        v['novelty_rewards'] = rewards

    def _update_rewards(self, metrics, v):
        self._update_skill_rewards(metrics, v)
        if self.use_kme and self.use_novelty_reward:
            self._update_distributional_novelty_rewards(metrics, v)
        rewards =  v['skill_rewards']  * v.get('novelty_rewards', 1)
        v['rewards'] = rewards
        metrics.update({
            'RewardMean': rewards.mean(),
            'RewardStd': rewards.std(),
        })

    def _normalize_sac_scalars(self, v):
        for key in ('rewards', 'dones'):
            value = v.get(key)
            if value is None or not torch.is_tensor(value):
                continue
            if value.dim() == 1:
                continue
            value = value.reshape(value.shape[0], -1)
            if value.shape[1] != 1:
                raise ValueError(f"Expected scalar {key} per transition, got shape={tuple(value.shape)}")
            v[key] = value[:, 0]

    def _update_external_rewards(self, metrics, v):
        rewards = v.get('rewards')
        if rewards is None:
            raise KeyError("Downstream training requires environment rewards in batch['rewards']")
        if not torch.is_tensor(rewards):
            rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        else:
            rewards = rewards.to(self.device).float()
        if rewards.dim() > 1:
            rewards = rewards.reshape(rewards.shape[0], -1)
            if rewards.shape[1] != 1:
                raise ValueError(f"Expected scalar environment rewards, got shape={tuple(rewards.shape)}")
            rewards = rewards[:, 0]
        v['rewards'] = rewards
        metrics.update({
            'RewardMean': rewards.mean(),
            'RewardStd': rewards.std(),
            'ExternalRewardMean': rewards.mean(),
            'ExternalRewardStd': rewards.std(),
            'PureRewardMean': rewards.detach().mean(),
            'PureRewardStd': rewards.detach().std(unbiased=False),
            'PureRewardMin': rewards.detach().min(),
            'PureRewardMax': rewards.detach().max(),
        })

    def _get_batch_emb_vectors(self, v):
        if self.use_kme and self.kernel_map_obs:
            return self._get_kernel_maps(v)
        return self._get_encoder_maps(v)

    def _get_encoder_maps(self, v):

        cur_z, next_z, skills = v.get('cur_z'), v.get('next_z'), v.get('skills')
        if cur_z is None:
            obs = v['obs']
            next_obs = v['next_obs']
            cur_z = self.traj_encoder(obs)
            next_z = self.traj_encoder(next_obs)
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        return v.get('cur_z'), v.get('next_z'), v.get('skills')

    def _get_kernel_maps(self, v):
        kernel_cur_z, kernel_next_z, kernel_skills = (
            v.get('kernel_cur_z'), v.get('kernel_next_z'), v.get('kernel_skills'))

        if  kernel_cur_z is None:
            traj_encoder = self.target_traj_encoder if self.use_target_traj_encoder else self.traj_encoder
            obs = v['obs']
            next_obs = v['next_obs']
            cur_z = traj_encoder(obs)
            next_z = traj_encoder(next_obs)
            kernel_cur_z = self.kernel(cur_z.mean) / sqrt(self.kernel.ensemble_size)
            kernel_next_z = self.kernel(next_z.mean) / sqrt(self.kernel.ensemble_size)
            # kernel_skills = v['skills'].repeat(1, self.kernel.ensemble_size) if self.discrete else self.kernel(v['skills']) / sqrt(self.kernel.ensemble_size)
            kernel_skills =  self.kernel(v['skills']) / sqrt(self.kernel.ensemble_size)
            v.update({
                'kernel_cur_z': kernel_cur_z,
                'kernel_next_z': kernel_next_z,
                'kernel_skills': kernel_skills,
                'cur_z': cur_z,
                'next_z': next_z,
            })
        return v.get('kernel_cur_z'), v.get('kernel_next_z'), v.get('kernel_skills')

    def _update_skill_rewards(self, metrics, v):
        # obs = v['obs']
        # next_obs = v['next_obs']
        cur_z, next_z, skills = self._get_batch_emb_vectors(v)
        if self.inner:
            target_z = next_z.mean - cur_z.mean

            if self.discrete:
                masks = (skills - skills.mean(dim=1, keepdim=True)) * self.dim_skill / (self.dim_skill - 1 if self.dim_skill != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * skills).sum(dim=1)
                rewards = inner

            # For dual objectives
        else:
            # target_dists = self.traj_encoder(next_obs) if not self.use_kme else self.traj_encoder(self.traj_encoder(next_obs))
            target_dists = next_z

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, skills.argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(skills)

        metrics.update({
            'SkillRewardMean': rewards.mean(),
            'SkillRewardStd': rewards.std(),
            'PureRewardMean': rewards.detach().mean(),
            'PureRewardStd': rewards.detach().std(unbiased=False),
            'PureRewardMin': rewards.detach().min(),
            'PureRewardMax': rewards.detach().max(),
        })
        delta_phi_norm = torch.linalg.vector_norm(
            _dist_or_tensor_mean(next_z) - _dist_or_tensor_mean(cur_z),
            dim=1,
        )
        metrics.update({
            'DeltaPhiNormMean': delta_phi_norm.detach().mean(),
            'DeltaPhiNormStd': delta_phi_norm.detach().std(unbiased=False),
            'DeltaPhiNormMax': delta_phi_norm.detach().max(),
        })

        v['skill_rewards'] = rewards

    def _update_loss_te(self, metrics, v):
        self._update_rewards(metrics, v) # compute self-supervised reward
        rewards = v['rewards']
        # print(f"skill rewards: {rewards.mean():.10f}")
        obs = v['obs']
        next_obs = v['next_obs']
        cur_z, next_z = v['cur_z'], v['next_z']
        metrics.update({
            'currentStateMean': torch.square(cur_z.mean).mean(),
            'currentStateStd': torch.norm(cur_z.mean).std(),
        })
        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            metrics.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x, phi_y, skills = self._get_batch_emb_vectors(v)

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                metrics.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            elif self.use_kme and self.dual_dist == 'skill_kme':
                cst_dist = 1e-6 * torch.einsum('ij,ij->i',(phi_x + phi_y) / 2,v['skill_kme']).unsqueeze(0)
            elif self.dual_dist == 'kernel_mmd':
                kernel_state, kernel_next_state, kernel_skills = self._get_kernel_maps(v)
                cst_dist = torch.square(kernel_next_state - kernel_state).mean(dim=1)
            elif self.dual_dist == 'kernel_sim_dist':
                kernel_state, kernel_next_state, kernel_skills = self._get_kernel_maps(v)
                cst_dist = 1 - (kernel_next_state * kernel_state).sum(dim=1)
            elif self.dual_dist == 'kernel_sim':
                kernel_state, kernel_next_state, kernel_skills = self._get_kernel_maps(v)
                cst_dist = (kernel_next_state * kernel_state).sum(dim=1)
            else:
                raise NotImplementedError
            metrics.update({
                "OriginalCstDist": cst_dist.mean(),
                "OriginalCstStd": cst_dist.std(),
            })
            if self.dual_dist != 'kernel_sim':
                cst_penalty = cst_dist - torch.square(phi_y.mean - phi_x.mean).mean(dim=1) # cst_penalty = dist(s, s') - ||\phi(s') - \phi(s) ||^2
            else:
                cst_penalty = (phi_y.mean - phi_x.mean).sum(dim=1) - cst_dist
            temporal_violation = cst_penalty
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack) # 做截断
            te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            metrics.update({
                'DualCstPenalty': cst_penalty.detach().mean(),
                'TemporalViolationMean': temporal_violation.detach().mean(),
                'TemporalViolationFrac': (temporal_violation.detach() > 0).float().mean(),
            })
        else:
            te_obj = rewards
            zero = rewards.detach().new_zeros(())
            dual_lam_param = getattr(getattr(self, 'dual_lam', None), 'param', None)
            dual_lam = dual_lam_param.detach().exp() if dual_lam_param is not None else zero
            metrics.update({
                'DualLam': dual_lam,
                'LossDualLam': zero,
                'DualCstPenalty': zero,
                'TemporalViolationMean': zero,
                'TemporalViolationFrac': zero,
            })

        loss_te = -te_obj.mean()

        metrics.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, metrics, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        metrics.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    #########################################################################################################
    #                                                                                                       #
    #                                        logging and eval                                               #
    #                                                                                                       #
    #########################################################################################################
    def setup_logger(self, log_dir):
        tabular_log_file = os.path.join(log_dir, 'progress.csv')
        text_log_file = os.path.join(log_dir, 'debug.log')
        tb_dir = os.path.join(log_dir, 'tb')

        self.writer = SummaryWriter(tb_dir)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DRQ_METRAAgent')
        handler = logging.FileHandler(text_log_file)
        self.logger.addHandler(handler)

        print('Logging to {}'.format(log_dir))

    def get_traj_states_from_processed(self, data, include_terminal=True):
        """
        data: agent.process_samples(paths) 的返回字典
        返回：list[np.ndarray]，每个元素是该轨迹的状态序列
             - include_terminal=True：形状 (T_i+1, ...)  含 s_T
             - include_terminal=False：形状 (T_i, ...)    仅 s_0..s_{T-1}
        """
        traj_states = []
        for obs, nxt in zip(data['obs'], data['next_obs']):
            if include_terminal:
                full = np.concatenate([obs, nxt[-1:]], axis=0)  # s0..s_T
            else:
                full = obs  # s0..s_{T-1}
            traj_states.append(full)
        return traj_states

    def _flatten_states(x):
        """
        (T, ...) → (T, D)；像素 uint8 会转 float32 并 /255.
        """
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        # 常见像素是 (T, C, H, W)；向量是 (T, D)
        return x.reshape(x.shape[0], -1).astype(np.float32)

    def _is_kitchen_env(self):
        try:
            name = getattr(self._env, 'name', None)
            if isinstance(name, str) and ('kitchen' in name.lower()):
                return True
        except Exception:
            pass
        try:
            clsname = self._env.__class__.__name__.lower()
            if 'kitchen' in clsname:
                return True
        except Exception:
            pass
        try:
            spaces = getattr(self._env, 'obs_space', None)
            if isinstance(spaces, dict) and ('success' in spaces):
                return True
        except Exception:
            pass
        return False

    def _evaluate_policy(self):
        if 'pre_training' == self.stage:
            self._evaluate_pretrain_policy()
        else:
            self._evaluate_finetune_policy()

    def _evaluate_finetune_policy(self):
        avg_ret = 0
        kitchen_successes = {}
        completed_tasks_counts = []

        num_episodes = 10
        extras = None
        if self.stage == 'finetune':
            skill = self.best_skill

            # 1. Run evaluation episodes using rollout_worker
            # Reuse _get_trajectories which uses rollout_worker and handles batching
            extras = self._generate_skill_extras(
                np.repeat(skill[None, :], num_episodes, axis=0)
            )

        trajectories = self._get_trajectories(
            batch_size=num_episodes,
            extras=extras,
            deterministic_policy=True
        )
        official_kitchen_metrics = {}
        if self._is_kitchen_env():
            official_kitchen_metrics = calc_kitchen_eval_metrics(trajectories)
        official_kitchen_metrics.update(self.coverage_tracker.compute_policy_metrics(trajectories))
        official_kitchen_metrics.update(self.coverage_tracker.compute_queue_metrics())
        official_kitchen_metrics.update(self.coverage_tracker.compute_total_metrics())

        # Discounted return
        with utils.GlobalContext({'phase': 'eval', 'policy': 'skill'}):
            perf = utils.log_performance_ex(
                self.step_itr,
                batch=TrajectoryBatch.from_trajectory_list(self._env_spec, trajectories),
                discount=self.discount,
                additional_records=official_kitchen_metrics,
            )
        avg_ret = np.mean(perf['undiscounted_returns'])
        self.writer.add_scalar('eval/undiscounted_return', avg_ret, self.step_itr)

        # For kitchen success tracking
        if self._is_kitchen_env():
            for path in trajectories:
                ep_completed_count = 0
                if 'env_infos' in path:
                    for k, v in path['env_infos'].items():
                        if 'success' in k and 'distance' not in k:
                            if k not in kitchen_successes:
                                kitchen_successes[k] = []
                            is_completed = v[-1] > 0.5
                            kitchen_successes[k].append(float(is_completed))
                            if is_completed:
                                ep_completed_count += 1
                completed_tasks_counts.append(ep_completed_count)

        # Log Eval Success Rates
        if self._is_kitchen_env():
            official_metrics = official_kitchen_metrics
            for key, value in official_metrics.items():
                self.writer.add_scalar(f'eval/{key}', value, self.step_itr)
            self.writer.add_scalar(
                'eval/kitchen/overall_6task_coverage',
                official_metrics['KitchenOverall'],
                self.step_itr,
            )
            self.writer.add_scalar(
                'eval/kitchen/policy_task_coverage',
                official_metrics['KitchenPolicyTaskCoverage'],
                self.step_itr,
            )
            self.writer.add_scalar(
                'eval/avg_completed_tasks',
                official_metrics['KitchenAvgCompletedTasksPerTraj'],
                self.step_itr,
            )

            for k, v in kitchen_successes.items():
                avg_success = np.mean(v)
                self.writer.add_scalar(f'eval/{k}_rate', avg_success, self.step_itr)
                self.logger.info(f"Step {self.step_itr}: Eval {k} Rate = {avg_success:.2f}")

            # Log Avg Completed Tasks
            if completed_tasks_counts:
                avg_completed = np.mean(completed_tasks_counts)
                self.writer.add_scalar('eval_legacy/avg_completed_tasks', avg_completed, self.step_itr)
                self.logger.info(f"Step {self.step_itr}: Eval Avg Completed Tasks = {avg_completed:.10f}")
                print(f"Step {self.step_itr}: Eval Avg Completed Tasks = {avg_completed:.10f}")

        self.logger.info(f"Step {self.step_itr}: Eval Discounted Return = {avg_ret}")
        print(f"Step {self.step_itr}: Eval Discounted Return = {avg_ret}")

        # 2. Record Video if requested
        if self.eval_record_video:
            self.logger.info("Recording video...")
            video_extras = {}
            if 'finetune' == self.stage:
                video_extras = self._generate_skill_extras(
                    np.repeat(skill[None, :],1, axis=0)
                )
            video_trajectories = self._get_trajectories(
                batch_size=1,
                extras=video_extras,
                deterministic_policy=True,
                state_record_pixeled=not self.use_encoder,
            )

            utils.record_video(
                self.snapshot_dir,
                self.step_itr,
                'video_finetune',
                video_trajectories,
                skip_frames=self.video_skip_frames,
                shape=(128, 128)
            )



    def _evaluate_pretrain_policy(self):
        if self.discrete:
            eye_skills = np.eye(self.dim_skill)
            random_skills = []
            colors = []
            for i in range(self.dim_skill):
                num_trajs_per_skill = self.num_random_trajectories // self.dim_skill + (i < self.num_random_trajectories % self.dim_skill)
                for _ in range(num_trajs_per_skill):
                    random_skills.append(eye_skills[i])
                    colors.append(i)
            random_skills = np.array(random_skills)
            colors = np.array(colors)
            num_evals = len(random_skills)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_skill <= 10 else 'tab20'
            random_skill_colors = []
            for i in range(num_evals):
                random_skill_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_skill_colors = np.array(random_skill_colors)
        else:
            random_skills = np.random.randn(self.num_random_trajectories, self.dim_skill)
            if self.unit_length:
                random_skills = random_skills / np.linalg.norm(random_skills, axis=1, keepdims=True)
            random_skill_colors = utils.get_skill_colors(random_skills * 4)
        '''
        随机生成轨迹? 
        '''
        random_trajectories = self._get_trajectories(
            batch_size=self.num_random_trajectories,
            extras=self._generate_skill_extras(random_skills),
            deterministic_policy=True,
        )

        if False: # TODO:
            with utils.FigManager(self.snapshot_dir, self.step_itr, 'TrajPlot_RandomZ', writer=self.writer, global_step=self.step_itr) as fm:
                self._env.render_trajectories(
                    random_trajectories, random_skill_colors, self.eval_plot_axis, fm.ax
                )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).float().to(self.device) for ob in data['obs']])
        # traj_states = self.get_traj_states_from_processed(data)

        skill_dists = self.traj_encoder(last_obs)

        skill_means = skill_dists.mean.detach().cpu().numpy()
        if self.inner:
            skill_stddevs = torch.ones_like(skill_dists.stddev.detach().cpu()).numpy()
        else:
            skill_stddevs = skill_dists.stddev.detach().cpu().numpy()
        skill_samples = skill_dists.mean.detach().cpu().numpy()

        skill_colors = random_skill_colors

        with utils.FigManager(self.snapshot_dir, self.step_itr, f'PhiPlot', writer=self.writer, global_step=self.step_itr) as fm: # PhiPlot just plots ϕ(s). The phi trajectories in the paper are also ϕ(s) trajectories from randomly sampled z's.
            utils.draw_2d_gaussians(skill_means, skill_stddevs, skill_colors, fm.ax)
            utils.draw_2d_gaussians(
                skill_samples,
                [[0.03, 0.03]] * len(skill_samples),
                skill_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_skill_metrics = {}

        # Videos
        if self.eval_record_video:
            print("Recording video.\n\n\n\n\n")
            if self.discrete:
                video_skills = np.eye(self.dim_skill)
                video_skills = video_skills.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_skill == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_skills = []
                    for angle in [3, 2, 1, 4]:
                        video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_skills.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_skills.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_skills = np.array(video_skills)
                else:
                    video_skills = np.random.randn(16, self.dim_skill)
                    if self.unit_length:
                        video_skills = video_skills / np.linalg.norm(video_skills, axis=1, keepdims=True)
                video_skills = video_skills.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                batch_size=len(video_skills),
                deterministic_policy=True,
                extras=self._generate_skill_extras(video_skills),
                state_record_pixeled=not self.use_encoder,
            )
            utils.record_video(self.snapshot_dir, self.step_itr, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames, shape=(128,128))

        eval_skill_metrics.update(self.calc_eval_metrics(random_trajectories, is_skill_trajectories=True))
        eval_skill_metrics.update(self.coverage_tracker.compute_policy_metrics(random_trajectories))
        eval_skill_metrics.update(self.coverage_tracker.compute_queue_metrics())
        eval_skill_metrics.update(self.coverage_tracker.compute_total_metrics())
        with utils.GlobalContext({'phase': 'eval', 'policy': 'skill'}):
            performance = utils.log_performance_ex(
                self.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_skill_metrics,
            )
            # Log performance metrics with 'eval/' prefix
            for k, v in performance['scalars'].items():
                self.writer.add_scalar('eval/' + k, v, self.step_itr)
            for k, v in performance['histograms'].items():
                self.writer.add_histogram('eval/' + k, v, self.step_itr)
        self._log_eval_metrics()

    def calc_eval_metrics(self, trajectories, is_skill_trajectories=True):
        eval_metrics = {}
        sum_returns = 0
        for traj in trajectories:
            sum_returns += traj['rewards'].sum()
        eval_metrics[f'ReturnOverall'] = sum_returns
        if self._is_kitchen_env():
            calc_env_metrics = getattr(self._env, 'calc_eval_metrics', None)
            if callable(calc_env_metrics):
                eval_metrics.update(calc_env_metrics(trajectories, is_option_trajectories=True))
            else:
                eval_metrics.update(calc_kitchen_eval_metrics(trajectories))
        else:
            calc_env_metrics = getattr(self._env, 'calc_eval_metrics', None)
            if callable(calc_env_metrics):
                eval_metrics.update(calc_env_metrics(trajectories, is_option_trajectories=True))

        return eval_metrics
    
    def log_diagnostics(self, pause_for_plot=False):
        total_time = (time.time() - self._start_time)
        self.logger.info('Time %.2f s' % total_time)
        epoch_time = (time.time() - self._itr_start_time)
        self.logger.info('EpochTime %.2f s' % epoch_time)
        self.writer.add_scalar('TotalEnvSteps', self.total_env_steps, self.total_epoch)
        self.writer.add_scalar('TotalEpoch', self.total_epoch, self.total_epoch)
        self.writer.add_scalar('TimeEpoch', epoch_time, self.total_epoch)
        self.writer.add_scalar('TimeTotal', total_time, self.total_epoch)
        self.writer.flush()

    def _log_eval_metrics(self):
        self.eval_log_diagnostics()
        self.plot_log_diagnostics()

    def eval_log_diagnostics(self):
        total_time = (time.time() - self._start_time)
        self.writer.add_scalar('eval/TotalEnvSteps', self.total_env_steps, self.step_itr)
        self.writer.add_scalar('eval/TotalEpoch', self.total_epoch, self.step_itr)
        self.writer.add_scalar('eval/TimeTotal', total_time, self.step_itr)
        self.writer.flush()

    def plot_log_diagnostics(self):
        self.writer.add_scalar('plot/TotalEnvSteps', self.total_env_steps, self.step_itr)
        self.writer.add_scalar('plot/TotalEpoch', self.total_epoch, self.step_itr)
        self.writer.flush()

    #########################################################################################################
    #                                                                                                       #
    #                                        save and restore model                                         #
    #                                                                                                       #
    #########################################################################################################
    def save(self, epoch, new_save=False, pt_save=False):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the runner is set up.

        """

        self.logger.info('Saving snapshot...')

        if new_save and epoch != 0:
            os.makedirs(os.path.join(self.snapshot_dir, f'models/epoch-{epoch}'), exist_ok=True)
            file_name = os.path.join(self.snapshot_dir, f'models/epoch-{epoch}/skill_policy.pt')
            torch.save({
                'discrete': self.discrete,
                'dim_skill': self.dim_skill,
                'policy': self.sac_trainer.skill_policy,
            }, file_name)
            file_name = os.path.join(self.snapshot_dir, f'models/epoch-{epoch}/traj_encoder.pt')
            if 'pre_training' == self.stage:
                torch.save({
                    'discrete': self.discrete,
                    'dim_skill': self.dim_skill,
                    'traj_encoder': self.traj_encoder,
                }, file_name)

        if pt_save and epoch != 0:
            os.makedirs(os.path.join(self.snapshot_dir, f'models/epoch-{epoch}'), exist_ok=True)
            file_name = os.path.join(self.snapshot_dir, f'models/epoch-{epoch}/skill_policy.pt')
            torch.save({
                'discrete': self.discrete,
                'dim_skill': self.dim_skill,
                'policy': self.sac_trainer.skill_policy,
            }, file_name)

        self.logger.info('Saved')

    def _calculate_dbi(self, mappings, labels, ensemble_size):
        """
        Davies-Bouldin Index optimized for linear kernel distance: d(M1,M2) = 1 - (M1.M2)/L
        s_i = 1 - ||Ci||^2/L, d_ij = 1 - (Ci.Cj)/L
        """
        if not isinstance(mappings, torch.Tensor):
            mappings = torch.from_numpy(mappings).to(self.device).float()

        unique_labels = torch.unique(labels)
        K = len(unique_labels)
        if K < 2: return 0.0

        # 1. Centroids and Distance Matrix D[i,j] = 1 - (Ci.Cj)/L
        centroids = torch.stack([mappings[labels == l].mean(0) for l in unique_labels])
        D = 1.0 - torch.matmul(centroids, centroids.T) / ensemble_size

        # 2. s_i are the diagonal elements, d_ij are the off-diagonal elements
        s = torch.diag(D)
        R = (s.view(-1, 1) + s.view(1, -1)) / torch.clamp(D, min=1e-8)
        R.fill_diagonal_(0)
        mean_R = torch.sum(R, dim=1) / (K - 1)

        return torch.mean(mean_R).item()

    def _calculate_ik_entropy(self, maps, ensemble_size):
        """
        Calculate state entropy using IKDE (Isolation Kernel Density Estimation).
        p(s) = (1/m) * <Phi(s), Phi_mean>
        Entropy H(S) = E[-log p(s)]
        """
        if not isinstance(maps, torch.Tensor):
            maps = torch.from_numpy(maps).to(self.device).float()

        # 1. Calculate average map (Phi_hat)
        phi_hat = maps.mean(dim=0) # (ensemble_size * subsample_size)

        # 2. Estimate density p(s) for each point
        # Inner product <Phi(s), Phi_hat> normalized by ensemble_size
        p_s = torch.sum(maps * phi_hat, dim=1) / ensemble_size

        # 3. Calculate Entropy
        entropy = -torch.mean(torch.log(p_s + 1e-9))
        return entropy.item()

    def _plot_trajectories(self):
        return _plot_trajectories(self)

    def _save_image_grid(self, trajectories, n_eval_skills, n_trajs_per_skill):
        """
        For each trajectory, sample 10 frames evenly, concatenate horizontally.
        Then concatenate all rows vertically, grouped by skill.
        Handles flattened observation pixels by reshaping them back to images.
        """
        import cv2
        all_rows = []
        n_frames = 64
        # Default target shape for MetaWorld/DMC renders in this codebase
        target_h, target_w = 128, 128

        for i in range(n_eval_skills):
            for j in range(n_trajs_per_skill):
                idx = i * n_trajs_per_skill + j
                if idx >= len(trajectories): continue

                traj = trajectories[idx]
                images = traj['observations']

                # Sample 10 frames evenly
                total_steps = len(images)
                indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)

                processed_frames = []
                for step_idx in indices:
                    img = images[step_idx]

                    # 1. Handle flattened 1D array
                    if img.ndim == 1:
                        # Try to reshape based on common sizes
                        if len(img) == target_h * target_w * 3:
                            img = img.reshape(target_h, target_w, 3)
                        else:
                            # Heuristic: assume square image with 3 channels
                            side = int(np.sqrt(len(img) // 3))
                            img = img.reshape(side, side, 3)

                    # 2. Handle (C, H, W) -> (H, W, C) if necessary
                    # In record_video, it transposes to (C, H, W) for saving video,
                    # but for grid concatenation we want (H, W, C).
                    if img.shape[0] == 3 and img.shape[1] > 3:
                        img = img.transpose(1, 2, 0)

                    processed_frames.append(img)

                # Horizontal concatenation for one trajectory
                row = np.concatenate(processed_frames, axis=1)
                all_rows.append(row)

        if not all_rows: return

        # Vertical concatenation for all trajectories
        final_grid = np.concatenate(all_rows, axis=0)

        # Save to disk
        save_path = os.path.join(self.snapshot_dir, 'plots', f'skill_grid_{self.step_itr}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert RGB to BGR for cv2
        grid_bgr = cv2.cvtColor(final_grid.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, grid_bgr)

        # Also log to TensorBoard
        if self.writer:
            # (H, W, C) -> (C, H, W) for TB
            tb_grid = final_grid.transpose(2, 0, 1)
            self.writer.add_image('eval/skill_grid', tb_grid, self.step_itr)
