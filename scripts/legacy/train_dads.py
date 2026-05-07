import warnings

from core.networks import GaussianMLPTwoHeadedModuleEx, get_gaussian_module_construction

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message='ing')
import numpy as np
import torch
import os
import time
import argparse


import tqdm
from utils import utils
from memory.replay_buffer import PathBufferEx
import dads # Import drq directly for DRQ_METRAAgent
import json
from envs import make_env


torch.backends.cudnn.benchmark = True

'''
# METRA on pixel-based car racing for debug
python train_metra.py --task debug_dummy --time_limit 50 --seed 0 --traj_batch_size 8 --video_skip_frames 2 --framestack 3 --sac_min_buffer_size 300 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 2 --n_epochs_per_log 5 --n_epochs_per_eval 5 --n_epochs_per_save 1 --n_epochs_per_pt_save 1 --discrete 0 --dim_skill 4 --encoder 1 --sample_cpu 0 --action_repeat 1 --n_epochs 10
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python train_metra.py --task metaworld_dial_turn --time_limit 250 --seed 0 --traj_batch_size 4 --video_skip_frames 4 --num_video_repeats 2 --framestack 3 --sac_max_buffer_size 100000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 100 --n_epochs_per_log 10 --n_epochs_per_eval 2000 --n_epochs_per_save 10000 --n_epochs_per_pt_save 10000 --discrete 1 --dim_skill 8 --encoder 1 --sample_cpu 0 --action_repeat 1 --n_epochs 200000
'''

class Workspace(object):
    """
    Manages the training and evaluation lifecycle for the DrQ(+METRA) agent.
    Handles environment creation, agent instantiation, replay buffer, logging, and video recording.
    Configuration is passed via an argparse Namespace.
    """
    def __init__(self, args):
        self.args = args
        # Create working directory for logs and outputs
        stage_dir = args.stage if hasattr(args, 'stage') and args.stage else 'unknown_stage'
        dist_dir = args.dual_dist if hasattr(args, 'dual_dist') else 'unknown_dual_dist'
        if args.algo == 'metra' and not args.inner and not args.dual_reg:
            dist_dir = 'diayn'
        encoder_stage = 'finetune_visual' if getattr(args, 'finetune_encoder', False) else 'freeze_visual'
        self.work_dir = f'./runs/{args.algo}/{args.task}/{stage_dir}/{encoder_stage}/{dist_dir}/{time.strftime("%Y%m%d-%H%M%S")}_seed{args.seed}'
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.work_dir + '/models', exist_ok=True)
        print(f'Workspace directory: {self.work_dir}')
        with open(os.path.join(self.work_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        utils.set_seed_everywhere(args.seed) # Set random seeds for reproducibility
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device (cuda/cpu)
        self.env = make_env(mode="train", config=args) # Create the DMC environment

        # Initialize Replay Buffer
        self.replay_buffer = PathBufferEx(capacity_in_transitions=int(args.sac_max_buffer_size),
                                          pixel_shape=self.env.obs_space['image'].shape)
        # Prepare parameters for DRQ_METRAAgent instantiation from args
        module_cls, module_kwargs = get_gaussian_module_construction(
            hidden_sizes=[args.model_master_dim] * args.model_master_num_layers,
            const_std=args.sd_const_std,
            hidden_nonlinearity=args.model_master_nonlinearity or torch.relu,
            input_dim=self.env.spec.observation_space.flat_dim + args.dim_skill,
            output_dim=self.env.spec.observation_space.flat_dim,
            min_std=0.3,
            max_std=10.0,
        )
        skill_dynamics = module_cls(**module_kwargs)
        agent_params = dict(
            env=self.env,
            tau=args.sac_tau,
            scale_reward=args.sac_scale_reward,
            target_coef=args.sac_target_coef,
            replay_buffer=self.replay_buffer,
            min_buffer_size=args.sac_min_buffer_size,
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
            pixel_shape=self.env.spec.observation_space.shape,
            env_name=args.task,
            algo=args.algo,
            env_spec=self.env.spec,
            skill_dynamics=skill_dynamics,
            dist_predictor=None,
            dual_lam=args.dual_lam,
            alpha=args.alpha,
            time_limit=args.time_limit,
            n_epochs_per_eval=args.n_epochs_per_eval,
            n_epochs_per_log=args.n_epochs_per_log,
            n_epochs_per_tb=args.n_epochs_per_log,
            n_epochs_per_save=args.n_epochs_per_save,
            n_epochs_per_pt_save=args.n_epochs_per_pt_save,
            n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
            dim_skill=args.dim_skill,
            num_random_trajectories=args.num_random_trajectories,
            num_video_repeats=args.num_video_repeats,
            eval_record_video=args.eval_record_video,
            video_skip_frames=args.video_skip_frames,
            eval_plot_axis=args.eval_plot_axis,
            name='METRA',
            device=self.device,
            sample_cpu=args.sample_cpu,
            num_train_per_epoch=1,
            sd_batch_norm=args.sd_batch_norm, # True, no use
            skill_dynamics_obs_dim=self.env.spec.observation_space.flat_dim, # no use
            trans_minibatch_size=args.trans_minibatch_size,
            trans_optimization_epochs=args.trans_optimization_epochs,
            discount=args.sac_discount,
            discrete=args.discrete,
            unit_length=args.unit_length,
            batch_size=args.traj_batch_size,
            snapshot_dir=self.work_dir,
            use_encoder=args.encoder,
            encoder_type=args.encoder_type,
            finetune_encoder=args.finetune_encoder,
            spectral_normalization=args.spectral_normalization,
            model_master_nonlinearity=args.model_master_nonlinearity,
            model_master_dim=args.model_master_dim,
            model_master_num_layers=args.model_master_num_layers,
            lr_op=args.lr_op,
            lr_te=args.lr_te,
            dual_lr=args.dual_lr,
            sac_lr_q=args.sac_lr_q,
            sac_lr_a=args.sac_lr_a,
            seed=args.seed,
            use_target_traj_encoder=args.use_target_traj_encoder,
            grad_clip_norm=args.grad_clip_norm,
            actor_init_std=args.actor_init_std,
            actor_max_log_std=args.actor_max_log_std,
            use_kme=args.use_kme,
            update_idk=args.update_idk,
            idk_subsample_size=args.idk_subsample_size,
            idk_init=args.idk_init,
            idk_from=args.idk_from,
            idk_groups=args.idk_groups,
            kernel_map = args.kernel_map,
            use_novelty_reward = args.use_novelty_reward,
            stage=args.stage,
            skill_policy_path=args.skill_policy_path,
            policy_delay=args.policy_delay,
            actor_start_steps=args.actor_start_steps,
            actor_critic_backbone=args.ac_backbone,
            simba_actor_hidden_dim=args.simba_actor_hidden_dim,
            simba_actor_num_blocks=args.simba_actor_num_blocks,
            simba_critic_hidden_dim=args.simba_critic_hidden_dim,
            simba_critic_num_blocks=args.simba_critic_num_blocks,
            simba_mlp_ratio=args.simba_mlp_ratio,
            simba_rsnorm_momentum=args.simba_rsnorm_momentum,
            simba_rsnorm_eps=args.simba_rsnorm_eps,
            simba_ln_eps=args.simba_ln_eps,

        )
        # Instantiate DRQ_METRAAgent with parameters derived from args
        self.agent = dads.DADS(**agent_params)
        self.agent.setup_logger(self.work_dir)  # Setup logging for the agent

    def run(self):
        """Main training loop."""
        self.agent.train(n_epochs=self.args.n_epochs)  # Start training the agent

        self.agent.save(epoch='final', pt_save=True)

def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument('--normalizer_type', type=str, default='off', choices=['off', 'preset'])
    parser.add_argument('--encoder', type=int, default=1)
    parser.add_argument('--encoder_type', type=str, default='original', choices=['original', 'resnet-101', 'dinov3'])
    parser.add_argument('--finetune_encoder', action='store_true', default=False)
    parser.add_argument('--task', type=str, default='dmc_walker_walk')
    parser.add_argument('--framestack', type=int, default=None)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--render_size', type=int, default=64)
    parser.add_argument('--flatten_obs', type=int, default=1, choices=[0, 1])
    parser.add_argument('--camera', type=str, default='corner')
    parser.add_argument('--dmc_camera', type=int, default=-1)

    parser.add_argument('--time_limit', type=int, default=200)

    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=4)
    parser.add_argument('--n_thread', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=1000000)
    parser.add_argument('--traj_batch_size', type=int, default=8)
    parser.add_argument('--trans_minibatch_size', type=int, default=256)
    parser.add_argument('--trans_optimization_epochs', type=int, default=200)

    parser.add_argument('--n_epochs_per_eval', type=int, default=125)
    parser.add_argument('--n_epochs_per_log', type=int, default=25)
    parser.add_argument('--n_epochs_per_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=1000)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=None)
    parser.add_argument('--num_random_trajectories', type=int, default=48)
    parser.add_argument('--num_video_repeats', type=int, default=2)
    parser.add_argument('--eval_record_video', type=int, default=1)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)

    parser.add_argument('--dim_skill', type=int, default=2)

    parser.add_argument('--common_lr', type=float, default=1e-4)
    parser.add_argument('--lr_op', type=float, default=None)
    parser.add_argument('--lr_te', type=float, default=None)

    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--algo', type=str, default='metra', choices=['metra', 'dads'])

    parser.add_argument('--sac_tau', type=float, default=5e-3)
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_min_buffer_size', type=int, default=10000)
    parser.add_argument('--sac_max_buffer_size', type=int, default=300000)
    parser.add_argument('--policy_delay', type=int, default=1, help="Delay policy updates by this factor")
    parser.add_argument('--actor_start_steps', type=int, default=0, help="Steps to warm up critic before updating actor")

    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])

    parser.add_argument('--model_master_dim', type=int, default=1024)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument('--model_master_nonlinearity', type=str, default=None, choices=['relu', 'tanh'])
    parser.add_argument('--sd_const_std', type=int, default=1)
    parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])

    parser.add_argument('--num_alt_samples', type=int, default=100)
    parser.add_argument('--split_group', type=int, default=65536)

    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1])
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1])
    parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1])  # Only for continuous skills

    parser.add_argument('--dual_reg', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=30)
    parser.add_argument('--dual_slack', type=float, default=1e-3)
    parser.add_argument('--dual_dist', type=str, default='one', choices=['l2', 's2_from_s', 'one',
                                                                         'skill_kme', 'kernel_mmd', 'kernel_sim_dist', 'kernel_sim'])
    parser.add_argument('--dual_lr', type=float, default=None)
    parser.add_argument('--use_kme', action="store_true", default=False,
                        help='whether use kernel mean embedding')
    parser.add_argument('--update_idk', type=int, default=1000,
                        help='rebuild IDK from replay every N updates (0 to disable)')
    parser.add_argument('--idk_subsample_size', type=int, default=256,
                        help='number of phi(s) to build IDK anchors')
    parser.add_argument('--idk_init', type=str, default='replay',
                        choices=['gaussian','uniform','replay'],
                        help='initialization of IDK basis in latent space')
    parser.add_argument('--idk_from', type=str, default='traj',
                        choices=['traj','enc'],
                        help='which latent to use for phi(s): traj encoder mean or pixel encoder feat')
    parser.add_argument('--idk_groups', type=int, default=1,
                        help='compute kernel mean with high samples by split to n groups')
    parser.add_argument('--kernel_map', action="store_true", default=False,
                        help='use kernel to map state encoder')
    parser.add_argument('--use_novelty_reward', action="store_true", default=False,
                        help='whether use kernel mean embedding')
    parser.add_argument('--use_target_traj_encoder', action="store_true", default=False,
                        help='whether use target trajectory encoder')
    parser.add_argument('--stage', type=str, default='pre_training', choices=['pre_training','finetune', 'zero_training'])
    parser.add_argument('--skill_policy_path', type=str)
    parser.add_argument('--grad_clip_norm', type=float, default=50.0)
    parser.add_argument('--actor_init_std', type=float, default=1.0)
    parser.add_argument('--actor_max_log_std', type=float, default=2.0)
    parser.add_argument('--ac_backbone', type=str, default='mlp', choices=['mlp', 'simba'])

    parser.add_argument('--simba_actor_hidden_dim', type=int, default=128)
    parser.add_argument('--simba_actor_num_blocks', type=int, default=1)
    parser.add_argument('--simba_critic_hidden_dim', type=int, default=512)
    parser.add_argument('--simba_critic_num_blocks', type=int, default=2)

    parser.add_argument('--simba_mlp_ratio', type=int, default=4)
    parser.add_argument('--simba_rsnorm_momentum', type=float, default=0.999)
    parser.add_argument('--simba_rsnorm_eps', type=float, default=1e-5)
    parser.add_argument('--simba_ln_eps', type=float, default=1e-5)

    return parser


def main():
    args = get_argparser().parse_args()

    # Correcting boolean arg parsing for store_true like behavior with defaults from YAML
    # The type=lambda x: (str(x).lower() == 'true') handles this for default=True cases.
    # If a flag like --no-save-video is preferred for a default True, then action='store_false' and dest='save_video' would be used.

    workspace = Workspace(args)
    workspace.run()


if __name__ == '__main__':
    main()
