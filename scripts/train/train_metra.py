import sys
import os
# Add src and root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

from utils import utils
from core.metra_config import get_parser, make_config_from_args, MetraConfig
from core.metra_agent import MetraAgent
from core.metra_trainer import MetraTrainer
from workspace_common import WorkspaceContext, build_env_and_replay_buffer, configure_runtime, save_args_json

configure_runtime()

class Workspace(object):
    """
    Manages the training and evaluation lifecycle for the DrQ(+METRA) agent.
    Handles environment creation, agent instantiation, replay buffer, logging, and video recording.
    """
    def __init__(self, args):
        self.args = args
        self.cfg = make_config_from_args(args)
        
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
        
        # Instantiate Agent
        self.agent = MetraAgent(self.cfg, self.env, self.replay_buffer)
        
        # Instantiate Trainer
        self.trainer = MetraTrainer(self.cfg, self.agent, self.env, self.replay_buffer, self.ctx.work_dir)
        if self.ctx.resume_checkpoint:
            self.trainer.load_resume_checkpoint(self.ctx.resume_checkpoint)

    def run(self):
        """Main training loop."""
        self.trainer.train()

def main():
    parser = get_parser()
    args = parser.parse_args()
    workspace = Workspace(args)
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
