
import sys
import os
# Add src and root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

from utils import utils

from core.metra_config import get_parser, make_config_from_args
from core.experimental.idk_csd_config import IdkCsdConfig
from core.experimental.idk_csd_agent import IdkCsdAgent
from core.experimental.idk_csd_trainer import IdkCsdTrainer
from workspace_common import WorkspaceContext, build_env_and_replay_buffer, configure_runtime, save_args_json

configure_runtime()

class IdkCsdWorkspace(object):
    """
    Workspace for IDK-CSD Experiment.
    """
    def __init__(self, args):
        self.args = args
        self.cfg = make_config_from_args(args, cls=IdkCsdConfig)
        
        # Initialize Context (paths, device)
        self.ctx = WorkspaceContext.create(self.cfg, resume_from=getattr(args, "resume_from", None))
        
        # Save arguments
        if self.ctx.is_resume:
            save_args_json(self.ctx.work_dir, args, filename='resume_args.json')
        else:
            save_args_json(self.ctx.work_dir, args)

        utils.set_seed_everywhere(self.cfg.seed)
        
        # Create Environment
        self.env, self.replay_buffer = build_env_and_replay_buffer(self.cfg, args)
        
        # Instantiate IDK-CSD Agent
        self.agent = IdkCsdAgent(self.cfg, self.env, self.replay_buffer)
        
        # Instantiate IDK-CSD Trainer
        self.trainer = IdkCsdTrainer(self.cfg, self.agent, self.env, self.replay_buffer, self.ctx.work_dir)
        if self.ctx.resume_checkpoint:
            self.trainer.load_resume_checkpoint(self.ctx.resume_checkpoint)

    def run(self):
        """Main training loop."""
        self.trainer.train()

def main():
    parser = get_parser()
    group = parser.add_argument_group('IDK-CSD')
    group.add_argument('--contrastive_n_epochs', type=int, default=5)
    group.add_argument('--contrastive_m_epochs', type=int, default=5)
    group.add_argument('--contrastive_warmup_epochs', type=int, default=5)
    group.add_argument('--contrastive_temperature', type=float, default=0.1)
    group.add_argument('--idk_update_interval', type=int, default=200)
    group.add_argument('--contrastive_rollout_batch_size', type=int, default=0)
    group.add_argument('--contrastive_temporal_budget', type=float, default=1.0)
    group.add_argument('--contrastive_mix_schedule', type=str, default='cosine', choices=['cosine', 'linear', 'exp'])
    group.add_argument('--contrastive_exp_k', type=float, default=5.0)
    group.add_argument('--traj_pos_encoding', type=str, default='rotary', choices=['rotary', 'sinusoidal', 'off'])
    group.add_argument('--traj_pos_encoding_base', type=float, default=10000.0)

    args, unknown = parser.parse_known_args()
    args.use_kme = True
    if '--dual_reg' not in sys.argv[1:]:
        args.dual_reg = 0

    workspace = IdkCsdWorkspace(args)
    try:
        workspace.run()
    finally:
        if hasattr(workspace, 'env') and workspace.env is not None:
             try:
                 workspace.env.close()
             except Exception:
                 pass

if __name__ == '__main__':
    main()
