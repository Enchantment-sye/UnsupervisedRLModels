import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm

from envs import make_env
from load_pretrain_metra import load_pretrained_metra
from sac_finetune import FinetuneSACAgent
from core.skill_selector import DiscreteSkillSelector, CEMSkillSelector
from memory.replay_buffer import PathBufferEx
from utils.checkpointing import infer_run_dir_from_checkpoint, resolve_resume_path, safe_torch_load

class Workspace:
    def __init__(self, args):
        self.args = args
        self.resume_checkpoint = None
        if args.resume_from:
            self.resume_checkpoint = resolve_resume_path(args.resume_from)
            self.work_dir = infer_run_dir_from_checkpoint(self.resume_checkpoint)
        else:
            self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        if self.resume_checkpoint:
            os.makedirs(self.work_dir, exist_ok=True)
        
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('TrainFinetuneSAC')
        
        # Env
        self.env = make_env(mode="train", config=args)
        self.env_spec = self.env.spec

        # Infer pixel_shape if not provided but use_encoder is True
        if args.use_encoder and args.pixel_shape is None:
             # Try accessing obs_space property first (common in custom wrappers like MyKitchenEnv)
             if hasattr(self.env, 'obs_space'):
                 obs_space = self.env.obs_space
             elif hasattr(self.env, 'observation_space'):
                 obs_space = self.env.observation_space
             else:
                 obs_space = None

             if obs_space is not None:
                 # Check if observation space is image-like
                 # Usually env.observation_space['image'] or similar
                 # make_env with framestack often returns a dict or Box
                 # Let's check env type
                 if isinstance(obs_space, dict):
                     if 'image' in obs_space:
                         # Handle akro.Box or gym.Box or numpy array
                         img_space = obs_space['image']
                         if hasattr(img_space, 'shape'):
                             args.pixel_shape = img_space.shape
                         elif isinstance(img_space, np.ndarray):
                             args.pixel_shape = img_space.shape
                 else:
                      # If it's a Box and looks like image (C, H, W)
                      if hasattr(obs_space, 'shape') and len(obs_space.shape) == 3:
                          args.pixel_shape = obs_space.shape
        
        # Load Pretrained Policy
        self.device = torch.device(args.device)
        self.skill_policy = None
        self.discrete_skill = args.discrete_skill
        
        if os.path.exists(args.skill_policy_path):
            self.pre_metra = load_pretrained_metra(
                os.path.dirname(args.skill_policy_path),
                device=self.device,
                skill_policy_name=os.path.basename(args.skill_policy_path),
                load_traj_encoder=False,
                freeze=False, 
                eval_mode=False
            )
            self.skill_policy = self.pre_metra.skill_policy
            
            # Determine dim_skill from loaded model
            if self.pre_metra.dim_skill:
                self.dim_skill = self.pre_metra.dim_skill
            self.discrete_skill = self.pre_metra.discrete
        else:
            self.logger.warning(f"Checkpoint not found at {args.skill_policy_path}! Initializing with RANDOM skill policy.")
            self.pre_metra = None
            self.skill_policy = None # FinetuneSACAgent will handle random initialization
            
            # Must provide dim_skill if random init
            if not args.dim_skill:
                raise ValueError("Checkpoint not found and --dim_skill not provided! Cannot initialize random policy.")
            self.dim_skill = args.dim_skill
        
        # Override dim_skill if explicitly provided (and matches or force override)
        if args.dim_skill:
            self.dim_skill = args.dim_skill
            
        if self.dim_skill is None:
             raise ValueError("Please provide --dim_skill or a valid checkpoint path")
        
        # Replay Buffer
        self.replay_buffer = PathBufferEx(
            capacity_in_transitions=args.sac_max_buffer_size,
            pixel_shape=args.pixel_shape
        )

        # Agent
        self.agent = FinetuneSACAgent(
            env=self.env,
            env_spec=self.env_spec,
            skill_policy=self.skill_policy, # Can be None now
            dim_skill=self.dim_skill,
            device=self.device,
            work_dir=self.work_dir,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            use_encoder=args.use_encoder,
            pixel_shape=args.pixel_shape,
            discrete_skill=self.discrete_skill,
            replay_buffer_capacity=args.sac_max_buffer_size,
            min_buffer_size=args.sac_min_buffer_size,
            batch_size=args.batch_size,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            video_skip_frames=args.video_skip_frames,
            traj_batch_size=args.traj_batch_size,
            trans_optimization_epochs=args.trans_optimization_epochs,
            print_step_reward=args.print_step_reward,
            replay_buffer=self.replay_buffer,
            time_limit=args.time_limit
        )
        if self.resume_checkpoint:
            checkpoint = safe_torch_load(self.resume_checkpoint, map_location=self.device)
            self.agent.load_resume_state(checkpoint)
            self.logger.info("Resumed fine-tune state from %s", self.resume_checkpoint)

    def find_best_skill(self):
        if self.discrete_skill:
            selector = DiscreteSkillSelector(
                env=self.env, 
                agent=self.agent, 
                device=self.device, 
                dim_skill=self.dim_skill,
                num_episodes=self.args.num_eval_episodes,
                logger=self.logger
            )
        else:
            selector = CEMSkillSelector(
                env=self.env, 
                agent=self.agent, 
                device=self.device, 
                dim_skill=self.dim_skill,
                cem_iters=self.args.cem_iters,
                cem_pop_size=self.args.cem_pop_size,
                cem_elites=self.args.cem_elites,
                cem_alpha=self.args.cem_alpha,
                update_mode=self.args.update_mode,
                cem_temperature=self.args.cem_temperature,
                cem_epsilon=self.args.cem_epsilon,
                num_episodes=self.args.num_eval_episodes,
                logger=self.logger
            )
            
        return selector.select()

    def run(self):
        # 1. Find Best Skill
        if self.agent.fixed_skill is None:
            best_skill = self.find_best_skill()
            self.agent.set_fixed_skill(best_skill)
        
        # 2. Start Training
        self.agent.train(n_epochs=self.args.n_epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--work_dir', type=str, default='finetune_results')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    
    # Env configuration (matched with train_metra.py)
    parser.add_argument('--render_size', type=int, default=64)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--camera', type=str, default='corner')
    parser.add_argument('--dmc_camera', type=int, default=-1)
    parser.add_argument('--time_limit', type=int, default=1000)
    parser.add_argument('--flatten_obs', type=int, default=1)
    parser.add_argument('--framestack', type=int, default=3)
    
    # Metra args
    parser.add_argument('--skill_policy_path', type=str, required=True)
    parser.add_argument('--dim_skill', type=int, default=None)
    parser.add_argument('--discrete_skill', action='store_true')
    parser.add_argument('--use_encoder', action='store_true')
    parser.add_argument('--pixel_shape', type=str, default=None) 
    
    # Training args
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--traj_batch_size', type=int, default=8)
    parser.add_argument('--trans_optimization_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--replay_buffer_capacity', type=int, default=1000000) # Deprecated
    parser.add_argument('--sac_max_buffer_size', type=int, default=1000000)
    parser.add_argument('--sac_min_buffer_size', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--num_eval_episodes', type=int, default=3, help='Number of episodes to eval each skill candidate')
    parser.add_argument('--log_freq', type=int, default=1) # Log every epoch
    parser.add_argument('--eval_freq', type=int, default=10) # Eval every 10 epochs
    parser.add_argument('--video_skip_frames', type=int, default=2)
    parser.add_argument('--print_step_reward', action='store_true', help="Print average episode reward to console every epoch")
    
    # CEM args
    parser.add_argument('--cem_iters', type=int, default=10)
    parser.add_argument('--cem_pop_size', type=int, default=100)
    parser.add_argument('--cem_elites', type=int, default=10)
    parser.add_argument('--cem_alpha', type=float, default=0.1, help='Smoothing factor for CEM updates (0 < alpha <= 1)')
    parser.add_argument('--update_mode', type=str, default='standard', choices=['standard', 'weighted'])
    parser.add_argument('--cem_temperature', type=float, default=1.0, help='Temperature for weighted update')
    parser.add_argument('--cem_epsilon', type=float, default=1e-5, help='Covariance regularization')

    args = parser.parse_args()
    
    if args.pixel_shape:
        args.pixel_shape = tuple(map(int, args.pixel_shape.split(',')))
    
    Workspace(args).run()
