import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import json
import argparse
from tqdm import tqdm

import torch
import numpy as np

from utils import utils
from utils.checkpointing import infer_run_dir_from_checkpoint, resolve_resume_path, safe_torch_load
from envs import make_env

# Put meta_controller_ppo_discrete.py next to this script (drq_url/)
from meta_controler import (
    DiscreteMetaControllerPPOTrainer,
    TrainerConfig,
    MetaPPOConfig,
)

from torch.utils.tensorboard import SummaryWriter
import numpy as np


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -------------------------
    # Run / logging
    # -------------------------
    parser.add_argument('--algo', type=str, default='meta_ppo_discrete')
    parser.add_argument('--run_group', type=str, default='Downstream')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--work_dir_root', type=str, default='./finetune')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')  # e.g. 'cuda', 'cuda:0', 'cpu'

    # -------------------------
    # Environment (must match envs.make_env expectations)
    # -------------------------
    parser.add_argument('--task', type=str, default='dmc_walker_walk',
                        help='Format: <suite>_<task>, e.g., dmc_walker_walk')
    parser.add_argument('--time_limit', type=int, default=200)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--render_size', type=int, default=64)
    parser.add_argument('--framestack', type=int, default=1)
    parser.add_argument('--flatten_obs', type=int, default=1, choices=[0, 1])
    parser.add_argument('--camera', type=str, default='corner')
    parser.add_argument('--dmc_camera', type=int, default=-1)

    # -------------------------
    # Pretrained low-level METRA policy checkpoint
    # -------------------------
    parser.add_argument('--skill_policy_path', type=str, required=True,
                        help='Path to pretrained METRA skill_policy.pt (saved by DRQ_METRAAgent.save())')
    parser.add_argument('--dim_skill', type=int, default=None,
                        help='Discrete skill count. If omitted, will try to read from skill_policy checkpoint.')

    # -------------------------
    # Meta-controller training (PPO on macro-steps)
    # -------------------------
    parser.add_argument('--k_steps', type=int, default=25,
                        help='Meta decision period: choose a skill every k steps')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor used both inside macro return and for macro-step discount gamma^k_eff')
    parser.add_argument('--total_env_steps', type=int, default=2000000,
                        help='Stop after collecting this many primitive env steps')
    parser.add_argument('--rollout_macro_steps', type=int, default=4096,
                        help='Collect this many macro-steps per PPO update')
    parser.add_argument('--low_level_deterministic', type=int, default=0, choices=[0, 1],
                        help='If low-level policy supports it, use deterministic/mode actions')

    # -------------------------
    # PPO hyperparameters
    # -------------------------
    parser.add_argument('--ppo_lr', type=float, default=3e-4)
    parser.add_argument('--ppo_clip_ratio', type=float, default=0.2)
    parser.add_argument('--ppo_train_iters', type=int, default=10)
    parser.add_argument('--ppo_minibatch_size', type=int, default=256)
    parser.add_argument('--ppo_vf_coef', type=float, default=0.5)
    parser.add_argument('--ppo_ent_coef', type=float, default=0.01)
    parser.add_argument('--ppo_gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_max_grad_norm', type=float, default=1.0)

    # -------------------------
    # Eval / save
    # -------------------------
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--eval_every_updates', type=int, default=5)
    parser.add_argument('--save_every_updates', type=int, default=10)

# -------------------------
# TensorBoard
# -------------------------
    parser.add_argument('--tb', type=int, default=1, choices=[0, 1], help='Enable TensorBoard logging')
    parser.add_argument('--tb_dirname', type=str, default='tb', help='Subdir under work_dir for TB logs')
    parser.add_argument('--tb_log_every_updates', type=int, default=1, help='Log TB scalars every N PPO updates')
    parser.add_argument('--tb_eval_video', type=int, default=1, choices=[0, 1], help='Write eval video to TB if available')


    return parser


class Workspace(object):
    """
    Mirrors train_metra.py structure:
    - Creates work dir
    - Builds env via envs.make_env
    - Loads pretrained METRA low-level skill policy (frozen)
    - Trains a discrete-skill PPO meta-controller
    """

    def __init__(self, args):
        self.args = args
        self.resume_checkpoint = None

        # Work directory
        if args.resume_from:
            self.resume_checkpoint = resolve_resume_path(args.resume_from)
            self.work_dir = infer_run_dir_from_checkpoint(self.resume_checkpoint)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.work_dir = os.path.join(
                args.work_dir_root,
                args.algo,
                args.task,
                f"{timestamp}_seed{args.seed}"
            )
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "models"), exist_ok=True)

        print(f"Workspace directory: {self.work_dir}")
        args_filename = "resume_args.json" if self.resume_checkpoint else "args.json"
        with open(os.path.join(self.work_dir, args_filename), "w") as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        # Seed
        utils.set_seed_everywhere(args.seed)

        # Device
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

        # Env
        self.env = make_env(mode="train", config=args)

        # Load pretrained low-level skill policy
        self.low_policy, ckpt_meta = self. _load_skill_policy(args.skill_policy_path)

        # Validate discrete skill space
        discrete = ckpt_meta.get("discrete", None)
        if discrete is not None and bool(discrete) is not True:
            raise ValueError(f"Expected discrete skill checkpoint, but ckpt has discrete={discrete}")

        # Determine dim_skill
        dim_skill = args.dim_skill
        if dim_skill is None:
            dim_skill = ckpt_meta.get("dim_skill", None)
        if dim_skill is None:
            raise ValueError("dim_skill is not provided and cannot be inferred from checkpoint. "
                             "Please pass --dim_skill.")
        self.dim_skill = int(dim_skill)

        # Build trainer
        trainer_cfg = TrainerConfig(
            k_steps=args.k_steps,
            gamma=args.gamma,
            max_episode_steps=None,  # env already has TimeLimit wrapper
            rollout_macro_steps=args.rollout_macro_steps,
            low_level_deterministic=bool(args.low_level_deterministic),
        )
        ppo_cfg = MetaPPOConfig(
            lr=args.ppo_lr,
            clip_ratio=args.ppo_clip_ratio,
            train_iters=args.ppo_train_iters,
            minibatch_size=args.ppo_minibatch_size,
            vf_coef=args.ppo_vf_coef,
            ent_coef=args.ppo_ent_coef,
            gae_lambda=args.ppo_gae_lambda,
            max_grad_norm=args.ppo_max_grad_norm,
        )

        self.trainer = DiscreteMetaControllerPPOTrainer(
            env=self.env,
            low_level_policy=self.low_policy,
            dim_skill=self.dim_skill,
            device=self.device,
            trainer_cfg=trainer_cfg,
            ppo_cfg=ppo_cfg,
        )

        self.update_idx = 0

        self.writer = None
        if args.tb:
            tb_dir = os.path.join(self.work_dir, args.tb_dirname)
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
            print(f"TensorBoard log dir: {tb_dir}")
        if self.resume_checkpoint:
            self._load_resume(self.resume_checkpoint)

    @staticmethod
    def _frames_to_tb_video(frames: list):
        """
        frames: list of HWC uint8 frames
        return: torch uint8 video tensor [1, T, C, H, W]
        """
        import torch
        import numpy as np

        if frames is None or len(frames) == 0:
            return None

        arr = np.asarray(frames)  # (T, H, W, 3)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            return None

        # (T, H, W, 3) -> (1, T, 3, H, W)
        arr = np.transpose(arr, (0, 3, 1, 2))
        arr = arr[None, ...]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        return torch.from_numpy(arr)


    def _load_skill_policy(self, path):
        """
        Accepts either:
          - torch.save({'policy': PolicyEx, 'dim_skill':..., 'discrete':...}, path)
          - torch.save(PolicyEx, path)
        Returns (policy_module, meta_dict).
        """
        obj = torch.load(path, map_location=self.device)

        meta = {}
        if isinstance(obj, dict):
            meta = dict(obj)
            if "policy" in obj and isinstance(obj["policy"], torch.nn.Module):
                policy = obj["policy"]
            elif "skill_policy" in obj and isinstance(obj["skill_policy"], torch.nn.Module):
                policy = obj["skill_policy"]
            else:
                raise ValueError(
                    f"Unsupported checkpoint dict format at {path}. "
                    f"Expected key 'policy' (module). Got keys={list(obj.keys())}"
                )
        elif isinstance(obj, torch.nn.Module):
            policy = obj
        else:
            raise ValueError(f"Unsupported checkpoint type: {type(obj)} at {path}")

        policy.to(self.device)
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)

        return policy, meta

    def _save(self, tag: str):
        out = {
            "meta_policy_state_dict": self.trainer.meta.state_dict(),
            "optimizer_state_dict": self.trainer.opt.state_dict(),
            "dim_skill": self.dim_skill,
            "k_steps": self.args.k_steps,
            "gamma": self.args.gamma,
            "total_env_steps": self.trainer.total_env_steps,
            "total_macro_steps": self.trainer.total_macro_steps,
            "update_idx": self.update_idx,
            "work_dir": self.work_dir,
        }
        path = os.path.join(self.work_dir, "models", f"meta_ppo_{tag}.pt")
        torch.save(out, path)
        torch.save(out, os.path.join(self.work_dir, "models", "latest_resume.pt"))
        print(f"[Save] {path}")

    def _load_resume(self, checkpoint_path):
        checkpoint = safe_torch_load(checkpoint_path, map_location=self.device)
        meta_state = checkpoint.get("meta_policy_state_dict")
        if meta_state is None:
            raise KeyError("Resume checkpoint is missing meta_policy_state_dict")
        self.trainer.meta.load_state_dict(meta_state)
        self.trainer.total_env_steps = int(checkpoint.get("total_env_steps", self.trainer.total_env_steps))
        self.trainer.total_macro_steps = int(checkpoint.get("total_macro_steps", self.trainer.total_macro_steps))
        self.update_idx = int(checkpoint.get("update_idx", self.update_idx))
        print(f"[Resume] {checkpoint_path}")

    def run(self):
        """
        Main training loop.
        We run one PPO update at a time so we can interleave eval/save similar to train_metra.py.
        """
        total_env_steps = int(self.args.total_env_steps)

        while self.trainer.total_env_steps < total_env_steps:
            # 1) Collect rollout_macro_steps macro-transitions + run PPO update
            prev_env_steps = self.trainer.total_env_steps
            buf, rollout_stats = self.trainer._rollout_until(self.args.rollout_macro_steps)
            update_stats = self.trainer._ppo_update(buf)
            self.update_idx += 1

            env_steps_this_update = self.trainer.total_env_steps - prev_env_steps

            if self.writer is not None and (self.update_idx % self.args.tb_log_every_updates == 0):
                gs = self.trainer.total_env_steps  # global_step: 用 env_steps 更直观

                # rollout stats
                self.writer.add_scalar("train/episode_return_mean", rollout_stats["episode_return_mean"], gs)
                self.writer.add_scalar("train/episode_return_std", rollout_stats.get("episode_return_std", 0.0), gs)
                self.writer.add_scalar("train/episode_steps_mean", rollout_stats.get("episode_steps_mean", 0.0), gs)

                # bookkeeping
                self.writer.add_scalar("train/env_steps", self.trainer.total_env_steps, gs)
                self.writer.add_scalar("train/macro_steps", self.trainer.total_macro_steps, gs)
                self.writer.add_scalar("train/env_steps_per_update", env_steps_this_update, gs)

                # PPO core stats
                self.writer.add_scalar("ppo/loss_pi", update_stats["loss_pi"], gs)
                self.writer.add_scalar("ppo/loss_v", update_stats["loss_v"], gs)
                self.writer.add_scalar("ppo/entropy", update_stats["entropy"], gs)
                self.writer.add_scalar("ppo/approx_kl", update_stats["approx_kl"], gs)

                # optional extra metrics if you add them in _ppo_update (下面第2部分会给)
                if "clipfrac" in update_stats:
                    self.writer.add_scalar("ppo/clipfrac", update_stats["clipfrac"], gs)
                if "explained_var" in update_stats:
                    self.writer.add_scalar("ppo/explained_var", update_stats["explained_var"], gs)
                if "grad_norm" in update_stats:
                    self.writer.add_scalar("ppo/grad_norm", update_stats["grad_norm"], gs)
                if "lr" in update_stats:
                    self.writer.add_scalar("ppo/lr", update_stats["lr"], gs)

                # skill 分布（如果 _rollout_until 返回了 macro_actions）
                if "macro_actions" in rollout_stats:
                    acts = np.asarray(rollout_stats["macro_actions"], dtype=np.int64)
                    self.writer.add_histogram("train/skill_idx", acts, gs)

            # 2) Log
            print(
                f"[MetaPPO] update={self.update_idx} env_steps={self.trainer.total_env_steps} "
                f"macro_steps={self.trainer.total_macro_steps} "
                f"ep_ret_mean={rollout_stats['episode_return_mean']:.10f} "
                f"loss_pi={update_stats['loss_pi']:.10f} loss_v={update_stats['loss_v']:.10f} "
                f"ent={update_stats['entropy']:.10f} kl={update_stats['approx_kl']:.10f}"
            )

            # 3) Eval
            # if self.args.eval_every_updates > 0 and (self.update_idx % self.args.eval_every_updates == 0):
            #     metrics = self.trainer.evaluate(num_episodes=self.args.eval_episodes)
            #     print(
            #         f"[Eval] update={self.update_idx} "
            #         f"return_mean={metrics['return_mean']:.3f} return_std={metrics['return_std']:.3f} "
            #         f"steps_mean={metrics['steps_mean']:.1f}"
            #     )

            if self.args.eval_every_updates > 0 and (self.update_idx % self.args.eval_every_updates == 0):
                video_dir = os.path.join(self.work_dir, "videos")
                self.trainer.evaluate_with_video(
                    num_episodes=16,
                    video_dir=video_dir,
                    video_tag=f"eval_u{self.update_idx}",
                )
                # print(
                #     f"[Eval] update={self.update_idx} "
                #     f"return_mean={metrics['return_mean']:.3f} return_std={metrics['return_std']:.3f} "
                #     f"avg_return_16={metrics['avg_return_16']:.3f} "
                #     f"avg_discounted_return_16={metrics['avg_discounted_return_16']:.3f} "
                #     f"steps_mean={metrics['steps_mean']:.1f} "
                #     f"video={metrics.get('video_path', 'None')}"
                # )

            # 4) Save
            if self.args.save_every_updates > 0 and (self.update_idx % self.args.save_every_updates == 0):
                self._save(tag=f"u{self.update_idx}")

        # Final save
        self._save(tag="final")
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()



def main():
    args = get_argparser().parse_args()
    workspace = Workspace(args)
    workspace.run()


if __name__ == "__main__":
    main()
