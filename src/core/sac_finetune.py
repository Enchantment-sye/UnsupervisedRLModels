import copy
import logging
import os
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict

from utils import utils
from core import sac_utils
from core.networks import ContinuousMLPQFunctionEx, Encoder, WithEncoder, GaussianMLPTwoHeadedModuleEx, PolicyEx
from memory.replay_buffer import PathBufferEx
from workers.rollout import SkillRolloutWorker
from data_structs.trajectory_batch import TrajectoryBatch

class FinetuneSACAgent:
    def __init__(self,
                 env,
                 env_spec,
                 skill_policy,
                 dim_skill,
                 device,
                 work_dir,
                 hidden_dim=1024,
                 num_layers=2,
                 lr=1e-4,
                 gamma=0.999,
                 tau=0.005,
                 alpha=0.1,
                 target_entropy=None,
                 use_encoder=False,
                 pixel_shape=None,
                 discrete_skill=False,
                 replay_buffer_capacity=1000000,
                 min_buffer_size=1000,
                 batch_size=256,
                 log_freq=10,
                 eval_freq=50,
                 video_skip_frames=2,
                 traj_batch_size=8,
                 trans_optimization_epochs=200,
                 print_step_reward=False,
                 replay_buffer=None,
                 time_limit=1000,
                 seed=1
                 ):
        self.env = env
        self.device = device
        self.work_dir = work_dir
        self.discount = gamma  # Renamed for sac_utils compatibility
        self.tau = tau
        self.dim_skill = dim_skill
        self.discrete_skill = discrete_skill
        self._env_spec = env_spec 
        self.replay_buffer_capacity = replay_buffer_capacity
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.video_skip_frames = video_skip_frames
        self.traj_batch_size = traj_batch_size
        self.trans_optimization_epochs = trans_optimization_epochs
        self.print_step_reward = print_step_reward
        self.pixel_shape = pixel_shape
        
        # Rollout Worker (handles env interaction and dict action wrapping)
        self.rollout_worker = SkillRolloutWorker(
            seed=seed, 
            time_limit=time_limit, 
            cur_extra_keys=['skill']
        )

        # Logger
        self.logger = logging.getLogger('FinetuneSAC')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()
        self.logger.addHandler(logging.FileHandler(os.path.join(work_dir, 'debug.log'), mode='a'))
        self.logger.addHandler(logging.StreamHandler())
        self.writer = SummaryWriter(work_dir)
        self.step_itr = 0
        self.start_epoch = 0
        self.last_epoch = -1

        # 1. Load Pre-trained Skill Policy
        self.use_encoder = use_encoder
        self.skill_policy = skill_policy
        
        # If skill_policy is None, initialize it randomly
        if self.skill_policy is None:
            self.logger.info("Initializing random skill policy...")
            
            # Helper to create encoder (copied logic from metra.py/sac_finetune.py context)
            def make_encoder():
                 return Encoder(pixel_shape=pixel_shape)

            if use_encoder:
                # Infer module_obs_dim
                example_encoder = make_encoder().to(device)
                with torch.no_grad():
                    dummy_obs = torch.zeros((1, *pixel_shape)).to(device)
                    # Encoder expects (B, C, H, W) usually, but here it seems to handle flattened?
                    # Let's check Encoder definition or usage.
                    # In sac_finetune.py usage: dummy_obs.reshape(1, -1) -> self.encoder(dummy_obs)
                    # If pixel_shape is set, Encoder is likely a CNN.
                    # In metra.py: example_encoder(torch.as_tensor(example_ob["image"]).float().unsqueeze(0))
                    # Assuming Encoder handles correct input shape.
                    
                    # For safety, let's just create the encoder and use it.
                    self.encoder = example_encoder
                    
                    # Flatten dummy to check output dim if needed, or rely on Encoder internals
                    # Let's assume Encoder works with (1, *pixel_shape) or flattened
                    # Based on L113: dummy_obs.reshape(1, -1)
                    dummy_flat = dummy_obs.reshape(1, -1)
                    module_obs_dim = self.encoder(dummy_flat).shape[-1]

                def with_encoder(module, encoder):
                    return WithEncoder(encoder=encoder, module=module)
            else:
                # Flat state
                module_obs_dim = env_spec.observation_space.flat_dim
                
                def with_encoder(module, encoder=None):
                    return module

            policy_q_input_dim = module_obs_dim + dim_skill
            action_dim = env_spec.action_space.flat_dim
            hidden_sizes = [hidden_dim] * num_layers
            
            # Create MLP module
            mlp_module = GaussianMLPTwoHeadedModuleEx(
                input_dim=policy_q_input_dim,
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=torch.nn.ReLU, # Default
                layer_normalization=False,
                max_std=np.exp(2.),
                normal_distribution_cls=utils.TanhNormal,
                output_w_init=functools.partial(utils.xavier_normal_ex, gain=1.),
                init_std=1.0,
            )
            
            if use_encoder:
                module = with_encoder(mlp_module, self.encoder)
            else:
                module = mlp_module

            self.skill_policy = PolicyEx(
                name='skill_policy',
                env_spec=env_spec,
                module=module,
                skill_info={'dim_skill': dim_skill}
            )

        self.skill_policy.to(device)
        self.skill_policy.train() 

        # Unlock parameters for fine-tuning
        for p in self.skill_policy.parameters():
            p.requires_grad = True

        
        # Calculate Q-function input dimension
        hidden_sizes = [hidden_dim] * num_layers
        action_dim = env_spec.action_space.flat_dim
        
        if use_encoder:
            # 1. Share encoder from skill_policy if available
            # Note: skill_policy is PolicyEx(module=WithEncoder(encoder, mlp))
            # So we need to access skill_policy._module.encoder
            if hasattr(self.skill_policy._module, 'encoder'):
                self.encoder = self.skill_policy._module.encoder
            else:
                # If loading failed to get encoder, create one (should not happen for pretrained)
                # But if we just created it above, we have it.
                if not hasattr(self, 'encoder'):
                    self.encoder = Encoder(pixel_shape=pixel_shape).to(device)


            # Create a separate encoder for Critic to avoid gradient conflict
            self.critic_encoder = copy.deepcopy(self.encoder).to(device)
            self.critic_encoder.train()
            for p in self.critic_encoder.parameters():
                p.requires_grad = True

            # Infer feature dim from the encoder
            with torch.no_grad():
                dummy_obs = torch.zeros((1, *pixel_shape)).to(device)
                # Flatten dummy_obs to match Encoder expectation (B, flattened_dim)
                dummy_obs = dummy_obs.reshape(1, -1)
                feature_dim = self.encoder(dummy_obs).shape[-1]
            
            q_input_dim = feature_dim + dim_skill
            
            # 3. Create Critics using the shared encoder
            # Important: The MLP part takes (feature + skill)
            # NOTE: We do NOT wrap with_shared_encoder here anymore. 
            # We handle encoding explicitly in update().
            
            self.qf1 = ContinuousMLPQFunctionEx(
                obs_dim=q_input_dim, 
                action_dim=action_dim,
                hidden_sizes=hidden_sizes
            ).to(device)
            
            self.qf2 = ContinuousMLPQFunctionEx(
                obs_dim=q_input_dim, 
                action_dim=action_dim,
                hidden_sizes=hidden_sizes
            ).to(device)
            
        else:
            obs_dim = env_spec.observation_space.flat_dim
            q_input_dim = obs_dim + dim_skill
            
            self.qf1 = ContinuousMLPQFunctionEx(
                obs_dim=q_input_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes
            ).to(device)
            
            self.qf2 = ContinuousMLPQFunctionEx(
                obs_dim=q_input_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes
            ).to(device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        # 3. Entropy / Alpha
        self._target_entropy = target_entropy if target_entropy is not None else -action_dim # Renamed for sac_utils
        self.log_alpha = utils.ParameterModule(torch.Tensor([np.log(alpha)])).to(device)

        # 4. Optimizers
        self.policy_optimizer = torch.optim.Adam(self.skill_policy.parameters(), lr=lr)
        
        # Critic optimizer includes critic_encoder parameters if using encoder
        critic_params = list(self.qf1.parameters()) + list(self.qf2.parameters())
        if use_encoder:
            critic_params += list(self.critic_encoder.parameters())
            
        self.qf_optimizer = torch.optim.Adam(critic_params, lr=lr)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=lr)
        
        # 5. Replay Buffer
        if replay_buffer is None:
            self.replay_buffer = PathBufferEx(
                capacity_in_transitions=replay_buffer_capacity,
                pixel_shape=self.pixel_shape
            )
        else:
            self.replay_buffer = replay_buffer

        self.fixed_skill = None
        
        self.skill_policy.train()
        self.target_qf1.train()
        self.target_qf2.train()

    def set_fixed_skill(self, skill):
        self.fixed_skill = torch.as_tensor(skill, device=self.device).float().unsqueeze(0)

    def get_resume_state(self, epoch):
        return {
            'format': 'finetune_sac_resume_v1',
            'work_dir': self.work_dir,
            'next_epoch': int(epoch) + 1,
            'step_itr': int(self.step_itr),
            'last_epoch': int(epoch),
            'dim_skill': int(self.dim_skill),
            'discrete_skill': bool(self.discrete_skill),
            'use_encoder': bool(self.use_encoder),
            'policy_state_dict': self.skill_policy.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'target_qf1_state_dict': self.target_qf1.state_dict(),
            'target_qf2_state_dict': self.target_qf2.state_dict(),
            'log_alpha_state_dict': self.log_alpha.state_dict(),
            'fixed_skill': None if self.fixed_skill is None else self.fixed_skill.detach().cpu(),
            'critic_encoder_state_dict': None if not self.use_encoder else self.critic_encoder.state_dict(),
        }

    def load_resume_state(self, checkpoint):
        policy_state = checkpoint.get('policy_state_dict')
        if policy_state is None:
            raise KeyError("Resume checkpoint is missing policy_state_dict")

        self.skill_policy.load_state_dict(policy_state)
        self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        self.target_qf1.load_state_dict(checkpoint['target_qf1_state_dict'])
        self.target_qf2.load_state_dict(checkpoint['target_qf2_state_dict'])
        self.log_alpha.load_state_dict(checkpoint['log_alpha_state_dict'])

        if self.use_encoder:
            critic_encoder_state = checkpoint.get('critic_encoder_state_dict')
            if critic_encoder_state is not None:
                self.critic_encoder.load_state_dict(critic_encoder_state)

        fixed_skill = checkpoint.get('fixed_skill')
        if fixed_skill is not None:
            self.fixed_skill = fixed_skill.to(self.device)

        self.step_itr = int(checkpoint.get('step_itr', self.step_itr))
        self.start_epoch = int(checkpoint.get('next_epoch', self.start_epoch))
        self.last_epoch = int(checkpoint.get('last_epoch', self.last_epoch))

    def act(self, obs, skill, sample=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device).float()
            skill = torch.as_tensor(skill, device=self.device).float()
            
            if obs.ndim == 1 or (obs.ndim == 3 and not self.use_encoder): 
                obs = obs.unsqueeze(0)
            if obs.ndim == 3 and self.use_encoder:
                 obs = obs.unsqueeze(0)
                 
            if skill.ndim == 1:
                skill = skill.unsqueeze(0)
            
            if self.use_encoder:
                 # Shared encoder handles obs -> feature
                 if hasattr(self.skill_policy._module, 'encoder'):
                     encoder = self.skill_policy._module.encoder
                     mlp = self.skill_policy._module.module
                     
                     obs_flat = obs.view(obs.shape[0], -1)
                     
                     feat = encoder(obs_flat)
                     
                     # 2. Concat skill
                     net_input = torch.cat([feat, skill], dim=-1)
                     
                     # 3. MLP Forward
                     dist = mlp(net_input)
                     if isinstance(dist, tuple):
                         dist = dist[0]
                 else:
                     raise RuntimeError("Skill policy does not have an encoder but use_encoder is True.")
            else:
                 # No encoder, direct concat
                 net_input = torch.cat([obs, skill], dim=-1)
                 dist = self.skill_policy(net_input)
                 if isinstance(dist, tuple):
                     dist = dist[0]
            
            if sample:
                action = dist.sample()
            else:
                action = dist.mean
                
            action_np = action.cpu().numpy()[0]
            # Align with wrappers: many envs expose .act_space dict and expect dict input to step()
            if hasattr(self.env, 'act_space'):
                # Use the first (and usually only) key from act_space, commonly 'action'
                try:
                    act_key = next(iter(self.env.act_space.keys()))
                except Exception:
                    act_key = 'action'
                return {act_key: action_np}
            return action_np

    def update(self, batch):
        obs = batch['obs']
        next_obs = batch['next_obs']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']
        
        B = obs.shape[0]
        skill = self.fixed_skill.repeat(B, 1)
        
        if not self.use_encoder:
             obs_with_skill = torch.cat([obs, skill], dim=-1)
             next_obs_with_skill = torch.cat([next_obs, skill], dim=-1)
        else:
             # Strategy:
             # We should NOT wrap qf1/qf2 with WithEncoder if we need to inject skill.
             # Instead, we should manually encode obs, concat skill, and pass to MLP.
             # This means qf1/qf2 should be the MLP parts.
             # And we handle encoding explicitly in update().
             pass

        # Here in update, assuming __init__ is fixed:
        if self.use_encoder:
            obs_enc_in = self._prepare_encoder_input(obs)
            next_obs_enc_in = self._prepare_encoder_input(next_obs)
            
            # Use critic_encoder for Q-function update
            obs_feat = self.critic_encoder(obs_enc_in)
            with torch.no_grad():
                next_obs_feat = self.critic_encoder(next_obs_enc_in)
            
            obs_with_skill = torch.cat([obs_feat, skill], dim=-1)
            next_obs_with_skill = torch.cat([next_obs_feat, skill], dim=-1)
            
        else:
            obs_with_skill = torch.cat([obs, skill], dim=-1)
            next_obs_with_skill = torch.cat([next_obs, skill], dim=-1)
        
        metrics = {}
        v = {}
        
        # Note: If use_encoder, qf1/qf2 are MLPs expecting (D+skill_dim).
        # sac_utils calls qf1(obs, action).
        
        if self.use_encoder:
            # Define MLPPolicyWrapper and policy_proxy early
            class MLPPolicyWrapper:
                def __init__(self, policy_ex):
                    self.policy_ex = policy_ex
                    # Assuming policy_ex.module is WithEncoder(encoder, mlp)
                    self.mlp = policy_ex._module.module
                
                def __call__(self, obs_with_skill):
    
                    dist = self.mlp(obs_with_skill)                   
                    return dist, {} # Return tuple (dist, info)
            
            policy_proxy = MLPPolicyWrapper(self.skill_policy)

            sac_utils.update_loss_qf(
                self, metrics, v,
                obs=obs_with_skill,
                actions=action,
                next_obs=next_obs_with_skill,
                dones=done,
                rewards=reward,
                policy=policy_proxy
            )
        else:
            sac_utils.update_loss_qf(
                self, metrics, v,
                obs=obs_with_skill,
                actions=action,
                next_obs=next_obs_with_skill,
                dones=done,
                rewards=reward,
                policy=self.skill_policy
            )
        
        # Optimize Q-function (Critic)
        self.qf_optimizer.zero_grad()
        (metrics['LossQf1'] + metrics['LossQf2']).backward()
        self.qf_optimizer.step()

        # For Policy update:
        # self.skill_policy is also (Encoder + MLP).
        # But here we are passing 'obs_with_skill' (features + skill).
        # If skill_policy expects raw obs, this fails.
        # If skill_policy expects features + skill, we need to bypass its encoder.
        
        # In __init__, we extracted self.encoder from skill_policy.
        # So skill_policy.module.module is the MLP.
        # We can pass a lambda/wrapper to sac_utils that uses the MLP directly.
        
        if self.use_encoder:
            
            # policy_proxy is already defined above
            
            # Update SAC Actor using the proxy and detached features
            # Note: We detach encoder features for actor update to prevent actor gradients messing up encoder
            # If we use separate encoders, we should use the actor's encoder (self.encoder) here!
            
            # Recalculate features using actor's encoder
            # self.encoder is shared with skill_policy (Actor)
            actor_obs_feat = self.encoder(obs_enc_in)
            # Detach is still good practice if we don't want actor loss to update encoder directly (standard DrQ)
            # But if we want to finetune actor encoder, we should NOT detach.
            # However, standard DrQ only updates encoder via Critic.
            # Let's follow standard DrQ: Actor uses detached features from its encoder (which is updated by Critic usually, but here they are separate).
            # WAIT: If they are separate, and we want to finetune, who updates self.encoder?
            # If only Critic updates self.critic_encoder, then self.encoder (Actor's) is never updated!
            # So Actor MUST update its own encoder if they are separate.
            
            # Decision: Since we separated them, Actor should update self.encoder.
            # So do NOT detach.
            
            actor_obs_with_skill = torch.cat([actor_obs_feat, skill], dim=-1)
            
            sac_utils.update_loss_sacp(
                self, metrics, v,
                obs=actor_obs_with_skill, # Use actor's encoder output
                policy=policy_proxy
            )
        else:
            sac_utils.update_loss_sacp(
                self, metrics, v,
                obs=obs_with_skill,
                policy=self.skill_policy
            )
        
        self.policy_optimizer.zero_grad()
        metrics['LossSacp'].backward()
        self.policy_optimizer.step()

        # Update Alpha
        sac_utils.update_loss_alpha(self, metrics, v)
        self.alpha_optimizer.zero_grad()
        metrics['LossAlpha'].backward()
        self.alpha_optimizer.step()
        
        # Update Targets
        sac_utils.update_targets(self)
        
        return metrics

    def _extract_obs(self, obs):
        if isinstance(obs, dict):
             if self.use_encoder:
                return obs['image']
             else:
                 if 'state' in obs: return obs['state']
                 if 'observation' in obs: return obs['observation']
                 for k, v in obs.items():
                     if k != 'image' and len(v.shape)==1: return v
        return obs
    
    def _prepare_encoder_input(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device).float()
        else:
            x = x.to(self.device).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        return x

    def _is_kitchen_env(self):
        try:
            name = getattr(self.env, 'name', None)
            if isinstance(name, str) and ('kitchen' in name.lower()):
                return True
        except Exception:
            pass
        try:
            clsname = self.env.__class__.__name__.lower()
            if 'kitchen' in clsname:
                return True
        except Exception:
            pass
        try:
            spaces = getattr(self.env, 'obs_space', None)
            if isinstance(spaces, dict) and ('success' in spaces):
                return True
        except Exception:
            pass
        return False

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self.batch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1:
                 value = np.squeeze(value, axis=1)
            
            data[key] = torch.from_numpy(value).float().to(self.device)
        
        # Remap keys to match update() expectation
        # PathBufferEx keys: obs, next_obs, actions, rewards, dones (usually)
        # update() expects: obs, next_obs, action, reward, done
        
        # Map plural to singular if needed, or update update()
        if 'actions' in data: data['action'] = data.pop('actions')
        if 'rewards' in data: data['reward'] = data.pop('rewards')
        if 'dones' in data: data['done'] = data.pop('dones')
        
        return data

    def collect_trajectories(self, num_trajs, skill):
        batches = []
        extras = [{'skill': skill} for _ in range(num_trajs)]
        
        # Use rollout_worker to collect trajectories
        # This handles dict action wrapping and other env details automatically
        for i in range(num_trajs):
            batch = self.rollout_worker.rollout(
                self.env, 
                self.skill_policy, 
                extra=extras[i], 
                deterministic_policy=False
            )
            batches.append(batch)
            
        trajectories = TrajectoryBatch.concatenate(*batches)
        # Convert to list of paths for buffer
        paths = trajectories.to_trajectory_list()
        
        # Post-process paths if needed (e.g. flattening obs if buffer expects flattened)
        # SkillRolloutWorker uses timestep['image'] which is likely (C,H,W) for pixel envs.
        # PathBufferEx with pixel_shape set might expect flattened or not.
        # sac_finetune.py previously flattened manually.
        # Let's check PathBufferEx.add_path.
        # It slices array[:len]. If array is (T, C, H, W), slicing works.
        # But if pixel_shape is set, PathBufferEx splits into pixel and state.
        # If obs is already flattened, it works.
        # If obs is (T, C, H, W), PathBufferEx logic:
        # self._buffer[pixel_key][start:stop] = array[... :pixel_dim]
        # This implies array MUST be flat (T, flat_dim) because it slices the last dimension!
        
        # So we MUST flatten observations here before returning.
        
        for path in paths:
            if self.pixel_shape is not None:
                if 'observations' in path and path['observations'].ndim > 2:
                     path['observations'] = path['observations'].reshape(path['observations'].shape[0], -1)
                if 'next_observations' in path and path['next_observations'].ndim > 2:
                     path['next_observations'] = path['next_observations'].reshape(path['next_observations'].shape[0], -1)
            
            # Ensure 2D for rewards/dones if needed (T, 1)
            if path['rewards'].ndim == 1: path['rewards'] = path['rewards'][:, None]
            if path['dones'].ndim == 1: path['dones'] = path['dones'][:, None]
            
        return paths

    def process_samples(self, paths):
        """
        Process paths to extract data for Replay Buffer, matching metra.py logic.
        """
        data = defaultdict(list)
        for path in paths:
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(utils.discount_cumsum(path['rewards'], self.discount))
            
            # Extract infos if needed, similar to metra.py
            if 'agent_infos' in path:
                if 'skill' in path['agent_infos']:
                     data['skills'].append(path['agent_infos']['skill'])

        return data

    def _update_replay_buffer(self, data):
        """
        Add processed data to Replay Buffer, matching metra.py logic.
        """
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if key in ['rewards', 'dones'] and cur_list.ndim == 1:
                         cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)
                self.step_itr += len(path['obs'])

    def train(self, n_epochs):
        if self.fixed_skill is None:
            raise ValueError("Fixed skill not set! Call set_fixed_skill first.")

        self.logger.info("Starting Fine-tuning (Epoch-based)...")
        best_skill_np = self.fixed_skill.cpu().numpy()[0]
        if self.start_epoch == 0 and self.step_itr == 0:
            self.step_itr = 0
        if self.start_epoch >= n_epochs:
            self.logger.info(
                "Resume start_epoch %d is already at or beyond configured n_epochs %d; nothing to do.",
                self.start_epoch,
                n_epochs,
            )
            return
        
        for epoch in tqdm(range(self.start_epoch, n_epochs)):
            # 1. Evaluate
            if epoch % self.eval_freq == 0:
                self.evaluate(epoch, best_skill_np, record_video=True)
                self.save(epoch)

            # 2. Collect Trajectories
            trajs = self.collect_trajectories(self.traj_batch_size, best_skill_np)
            
            # 3. Add to Buffer (metra-style)
            # Use process_samples to extract data from trajectories
            path_data = self.process_samples(trajs)
            # Use _update_replay_buffer to add to buffer
            self._update_replay_buffer(path_data)

            # 4. Update
            metrics = None
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                for _ in range(self.trans_optimization_epochs):
                    if self.replay_buffer.n_transitions_stored > self.batch_size:
                        batch = self._sample_replay_buffer()
                        metrics = self.update(batch)
            else:
                self.logger.info(f"Skipping update: Buffer size {self.replay_buffer.n_transitions_stored} < {self.min_buffer_size}")
            
            # 5. Log
            if epoch % self.log_freq == 0:
                if metrics:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'train/{k}', v, epoch)
                    
                    with torch.no_grad():
                        total_norm = utils.compute_total_norm(list(self.skill_policy.parameters()) + 
                                                              list(self.qf1.parameters()) + 
                                                              list(self.qf2.parameters()) + 
                                                              list(self.log_alpha.parameters()))
                        self.writer.add_scalar('train/TotalGradNorm', total_norm.item(), epoch)
                
                # Log Discounted Episode Return (using data from process_samples)
                # path_data['returns'] contains discounted returns for each trajectory
                # We take the first element (return at t=0) of each return array
                avg_ep_discounted_return = np.mean([ret[0] for ret in path_data['returns']])
                self.writer.add_scalar('train/discounted_return', avg_ep_discounted_return, epoch)
                
                # Print Step Reward to console if requested
                if self.print_step_reward:
                    print(f"Epoch {epoch}: Avg Ep Discounted Return = {avg_ep_discounted_return:.10f}")

            # Log Kitchen Success Rates (Train) - Log every epoch if it's kitchen env
            if self._is_kitchen_env():
                # 1. Track success rate per task (already done)
                kitchen_successes = {}
                # 2. Track number of completed tasks per trajectory
                completed_tasks_counts = []
                
                for traj in trajs:
                    traj_completed_count = 0
                    # Identify all task keys first
                    task_keys = [k for k in traj.keys() if 'success' in k]
                    
                    for k in task_keys:
                        if k not in kitchen_successes:
                            kitchen_successes[k] = []
                        # Check if task was completed at the end of trajectory
                        is_completed = traj[k][-1] > 0.5 # Assuming 1.0 is success
                        kitchen_successes[k].append(float(is_completed))
                        if is_completed:
                            traj_completed_count += 1
                    
                    completed_tasks_counts.append(traj_completed_count)
                
                # Log per-task success rate
                for k, v in kitchen_successes.items():
                    avg_success = np.mean(v)
                    self.writer.add_scalar(f'train/{k}_rate', avg_success, epoch)
                
                # Log average number of completed tasks
                avg_completed_tasks = np.mean(completed_tasks_counts)
                self.writer.add_scalar('train/avg_completed_tasks', avg_completed_tasks, epoch)
            self.last_epoch = epoch

    def evaluate(self, step, skill, record_video=False):
        avg_ret = 0
        kitchen_successes = {}
        completed_tasks_counts = []
        
        num_episodes = 10
        
        batches = []
        extras = [{'skill': skill} for _ in range(num_episodes)]
        
        # 1. Run evaluation episodes using rollout_worker
        for i in range(num_episodes):
            batch = self.rollout_worker.rollout(
                self.env, 
                self.skill_policy, 
                extra=extras[i], 
                deterministic_policy=True
            )
            batches.append(batch)
            
        # Combine batches to iterate easily
        trajectories = TrajectoryBatch.concatenate(*batches).to_trajectory_list()
        
        # Ensure 'observations' key exists for TrajectoryBatch.from_trajectory_list
        # If collect_trajectories logic (or rollout_worker) changed keys to 'obs', we need to restore them for logging
        for t in trajectories:
            if 'obs' in t and 'observations' not in t:
                t['observations'] = t['obs']
            if 'next_obs' in t and 'next_observations' not in t:
                t['next_observations'] = t['next_obs']

        # Discounted return (metra-style)
        with utils.GlobalContext({'phase': 'eval', 'policy': 'skill'}):
            perf = utils.log_performance_ex(
                step,
                batch=TrajectoryBatch.from_trajectory_list(self._env_spec, trajectories),
                discount=self.discount,
            )
        avg_ret = np.mean(perf['discounted_returns'])
        self.writer.add_scalar('eval/discounted_return', avg_ret, step)
        
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
             for k, v in kitchen_successes.items():
                 avg_success = np.mean(v)
                 self.writer.add_scalar(f'eval/{k}_rate', avg_success, step)
                 self.logger.info(f"Step {step}: Eval {k} Rate = {avg_success:.2f}")
             
             # Log Avg Completed Tasks
             if completed_tasks_counts:
                 avg_completed = np.mean(completed_tasks_counts)
                 self.writer.add_scalar('eval/avg_completed_tasks', avg_completed, step)
                 self.logger.info(f"Step {step}: Eval Avg Completed Tasks = {avg_completed:.2f}")

        self.logger.info(f"Step {step}: Eval Discounted Return = {avg_ret}")
        
        # 2. Record Video if requested
        if record_video:
            # We can reuse one of the trajectories for video if it has observations
            # Or generate a new one if we want specific behavior.
            # metra.py uses utils.record_video which handles reconstruction.
            # Let's generate a new one to match previous logic (separate call)
            
            self.logger.info("Recording video...")
            video_batch = self.rollout_worker.rollout(
                self.env, 
                self.skill_policy, 
                extra={'skill': skill}, 
                deterministic_policy=True
            )
            # Convert to list of dicts as expected by utils.record_video
            video_trajs = video_batch.to_trajectory_list()
            
            utils.record_video(
                self.work_dir, 
                step, 
                'video', 
                video_trajs, 
                skip_frames=self.video_skip_frames,
                shape=(128, 128)  # metra default 128
            )



    def save(self, step):
        path = os.path.join(self.work_dir, f'model_{step}.pt')
        torch.save(self.skill_policy.state_dict(), path)
        resume_state = self.get_resume_state(step)
        torch.save(resume_state, os.path.join(self.work_dir, 'latest_resume.pt'))
        torch.save(resume_state, os.path.join(self.work_dir, f'resume_state_{step}.pt'))
