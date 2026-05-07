import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="ing")

import os
import time
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal, List

import torch
from utils import utils
from memory.replay_buffer import PathBufferEx
import metra
from envs import make_env

torch.backends.cudnn.benchmark = True


# =========================
# 1) Config dataclasses
# =========================

@dataclass(frozen=True)
class RunDirConfig:
    
    log_dir: str

    def compute_tags(self, args: argparse.Namespace) -> Dict[str, Any]:
        stage_dir = args.stage if getattr(args, "stage", None) else "unknown_stage"
        dist_dir = args.dual_dist if getattr(args, "dual_dist", None) else "unknown_dual_dist"
        psi = args.idk_subsample_size if hasattr(args, "idk_subsample_size") else None

        # 原逻辑：algo==metra 且 not inner 且 not dual_reg -> dist_dir=diayn
        if args.algo == "metra" and (not args.inner) and (not args.dual_reg):
            dist_dir = "diayn"

        encoder_stage = "finetune_visual" if getattr(args, "finetune_encoder", False) else "freeze_visual"
        return dict(stage_dir=stage_dir, dist_dir=dist_dir, psi=psi, encoder_stage=encoder_stage)

    def build(self, args: argparse.Namespace) -> Path:
        tags = self.compute_tags(args)
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_seed{args.seed}"

        work_dir = Path(self.log_dir) / args.task / tags["stage_dir"] / tags["encoder_stage"] / str(tags["dist_dir"]) / str(tags["psi"]) / run_name
        (work_dir / "models").mkdir(parents=True, exist_ok=True)

        print(f"Workspace directory: {work_dir}")
        with (work_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        return work_dir


@dataclass(frozen=True)
class ComputeConfig:

    use_gpu: int
    sample_cpu: int

    def device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


@dataclass(frozen=True)
class ScheduleConfig:

    n_epochs: int
    time_limit: int


@dataclass(frozen=True)
class LoggingConfig:

    n_epochs_per_eval: int
    n_epochs_per_log: int
    n_epochs_per_save: int
    n_epochs_per_pt_save: int
    n_epochs_per_pkl_update: Optional[int]

    def resolved_pkl_update(self) -> int:
        # 原逻辑：None -> n_epochs_per_eval
        return self.n_epochs_per_eval if self.n_epochs_per_pkl_update is None else self.n_epochs_per_pkl_update


@dataclass(frozen=True)
class RolloutConfig:

    traj_batch_size: int


@dataclass(frozen=True)
class UpdateConfig:

    trans_minibatch_size: int
    trans_optimization_epochs: int
    num_train_per_epoch: int = 1  # 原代码写死 1


@dataclass(frozen=True)
class SACConfig:

    tau: float
    discount: float
    scale_reward: float
    target_coef: float
    alpha: float  # moved here

    sac_lr_q: Optional[float]
    sac_lr_a: Optional[float]
    min_buffer_size: int
    max_buffer_size: int

    policy_delay: int
    actor_start_steps: int


@dataclass(frozen=True)
class DualConfig:

    algo: Literal["metra", "dads"]
    inner: int
    num_alt_samples: int
    split_group: int

    dual_reg: int
    dual_lam: float
    dual_slack: float
    dual_dist: str
    dual_lr: Optional[float]


@dataclass(frozen=True)
class SkillConfig:

    dim_skill: int
    discrete: int
    unit_length: int

    stage: str
    skill_policy_path: Optional[str]


@dataclass(frozen=True)
class EncoderConfig:

    use_encoder: int
    encoder_type: str
    finetune_encoder: bool

    spectral_normalization: int
    model_master_dim: int
    model_master_num_layers: int
    model_master_nonlinearity: Optional[str]

    lr_op: Optional[float]
    lr_te: Optional[float]

    sd_batch_norm: int  # 原注释写 no use，但保持传参一致


@dataclass(frozen=True)
class PolicyConfig:

    grad_clip_norm: float
    actor_init_std: float
    actor_max_log_std: float


@dataclass(frozen=True)
class NoveltyConfig:

    use_kme: bool
    update_idk: int
    idk_subsample_size: int
    idk_init: str
    idk_from: str
    idk_groups: int
    kernel_map: bool
    use_novelty_reward: bool
    use_target_traj_encoder: bool


@dataclass(frozen=True)
class BackboneConfig:

    actor_critic_backbone: str

    simba_actor_hidden_dim: int
    simba_actor_num_blocks: int
    simba_critic_hidden_dim: int
    simba_critic_num_blocks: int

    simba_mlp_ratio: int
    simba_rsnorm_momentum: float
    simba_rsnorm_eps: float
    simba_ln_eps: float


@dataclass(frozen=True)
class EvaluationConfig:

    num_random_trajectories: int
    num_video_repeats: int
    eval_record_video: int
    video_skip_frames: int
    eval_plot_axis: Optional[List[float]]


@dataclass(frozen=True)
class ExperimentConfig:

    task: str
    seed: int

    run_dir: RunDirConfig
    compute: ComputeConfig
    schedule: ScheduleConfig
    logging: LoggingConfig
    rollout: RolloutConfig
    update: UpdateConfig

    sac: SACConfig
    dual: DualConfig
    skill: SkillConfig

    encoder: EncoderConfig
    policy: PolicyConfig
    novelty: NoveltyConfig
    backbone: BackboneConfig
    evaluation: EvaluationConfig  # renamed

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ExperimentConfig":
        log_dir = getattr(args, "log_dir", os.environ.get("METRA_LOG_DIR", "/share/shangyy"))

        return ExperimentConfig(
            task=args.task,
            seed=args.seed,

            run_dir=RunDirConfig(log_dir=log_dir),
            compute=ComputeConfig(use_gpu=args.use_gpu, sample_cpu=args.sample_cpu),
            schedule=ScheduleConfig(n_epochs=args.n_epochs, time_limit=args.time_limit),
            logging=LoggingConfig(
                n_epochs_per_eval=args.n_epochs_per_eval,
                n_epochs_per_log=args.n_epochs_per_log,
                n_epochs_per_save=args.n_epochs_per_save,
                n_epochs_per_pt_save=args.n_epochs_per_pt_save,
                n_epochs_per_pkl_update=args.n_epochs_per_pkl_update,
            ),
            rollout=RolloutConfig(traj_batch_size=args.traj_batch_size),
            update=UpdateConfig(
                trans_minibatch_size=args.trans_minibatch_size,
                trans_optimization_epochs=args.trans_optimization_epochs,
                num_train_per_epoch=1,
            ),

            sac=SACConfig(
                tau=args.sac_tau,
                discount=args.sac_discount,
                scale_reward=args.sac_scale_reward,
                target_coef=args.sac_target_coef,
                alpha=args.alpha,  # moved to SAC
                sac_lr_q=args.sac_lr_q,
                sac_lr_a=args.sac_lr_a,
                min_buffer_size=args.sac_min_buffer_size,
                max_buffer_size=args.sac_max_buffer_size,
                policy_delay=args.policy_delay,
                actor_start_steps=args.actor_start_steps,
            ),

            dual=DualConfig(
                algo=args.algo,
                inner=args.inner,
                num_alt_samples=args.num_alt_samples,
                split_group=args.split_group,
                dual_reg=args.dual_reg,
                dual_lam=args.dual_lam,
                dual_slack=args.dual_slack,
                dual_dist=args.dual_dist,
                dual_lr=args.dual_lr,
            ),

            skill=SkillConfig(
                dim_skill=args.dim_skill,
                discrete=args.discrete,
                unit_length=args.unit_length,
                stage=args.stage,
                skill_policy_path=args.skill_policy_path,
            ),

            encoder=EncoderConfig(
                use_encoder=args.encoder,
                encoder_type=args.encoder_type,
                finetune_encoder=args.finetune_encoder,
                spectral_normalization=args.spectral_normalization,
                model_master_nonlinearity=args.model_master_nonlinearity,
                model_master_dim=args.model_master_dim,
                model_master_num_layers=args.model_master_num_layers,
                lr_op=args.lr_op,
                lr_te=args.lr_te,
                sd_batch_norm=args.sd_batch_norm,
            ),

            policy=PolicyConfig(
                grad_clip_norm=args.grad_clip_norm,
                actor_init_std=args.actor_init_std,
                actor_max_log_std=args.actor_max_log_std,
            ),

            novelty=NoveltyConfig(
                use_kme=args.use_kme,
                update_idk=args.update_idk,
                idk_subsample_size=args.idk_subsample_size,
                idk_init=args.idk_init,
                idk_from=args.idk_from,
                idk_groups=args.idk_groups,
                kernel_map=args.kernel_map,
                use_novelty_reward=args.use_novelty_reward,
                use_target_traj_encoder=args.use_target_traj_encoder,
            ),

            backbone=BackboneConfig(
                actor_critic_backbone=args.ac_backbone,
                simba_actor_hidden_dim=args.simba_actor_hidden_dim,
                simba_actor_num_blocks=args.simba_actor_num_blocks,
                simba_critic_hidden_dim=args.simba_critic_hidden_dim,
                simba_critic_num_blocks=args.simba_critic_num_blocks,
                simba_mlp_ratio=args.simba_mlp_ratio,
                simba_rsnorm_momentum=args.simba_rsnorm_momentum,
                simba_rsnorm_eps=args.simba_rsnorm_eps,
                simba_ln_eps=args.simba_ln_eps,
            ),

            evaluation=EvaluationConfig(
                num_random_trajectories=args.num_random_trajectories,
                num_video_repeats=args.num_video_repeats,
                eval_record_video=args.eval_record_video,
                video_skip_frames=args.video_skip_frames,
                eval_plot_axis=args.eval_plot_axis,
            ),
        )

    def to_agent_kwargs(self, *, env, replay_buffer, work_dir: Path, device: torch.device) -> Dict[str, Any]:
        obs_dim = env.spec.observation_space.flat_dim if self.encoder.use_encoder else 0

        return dict(
            env=env,
            tau=self.sac.tau,
            scale_reward=self.sac.scale_reward,
            target_coef=self.sac.target_coef,
            replay_buffer=replay_buffer,
            min_buffer_size=self.sac.min_buffer_size,

            inner=self.dual.inner,
            num_alt_samples=self.dual.num_alt_samples,
            split_group=self.dual.split_group,
            dual_reg=self.dual.dual_reg,
            dual_slack=self.dual.dual_slack,
            dual_dist=self.dual.dual_dist,

            pixel_shape=env.spec.observation_space.shape,
            env_name=self.task,
            algo=self.dual.algo,
            env_spec=env.spec,

            skill_dynamics=None,
            dist_predictor=None,

            dual_lam=self.dual.dual_lam,
            alpha=self.sac.alpha,  # SAC alpha

            time_limit=self.schedule.time_limit,
            n_epochs_per_eval=self.logging.n_epochs_per_eval,
            n_epochs_per_log=self.logging.n_epochs_per_log,
            n_epochs_per_tb=self.logging.n_epochs_per_log,
            n_epochs_per_save=self.logging.n_epochs_per_save,
            n_epochs_per_pt_save=self.logging.n_epochs_per_pt_save,
            n_epochs_per_pkl_update=self.logging.resolved_pkl_update(),

            dim_skill=self.skill.dim_skill,
            num_random_trajectories=self.evaluation.num_random_trajectories,
            num_video_repeats=self.evaluation.num_video_repeats,
            eval_record_video=self.evaluation.eval_record_video,
            video_skip_frames=self.evaluation.video_skip_frames,
            eval_plot_axis=self.evaluation.eval_plot_axis,

            name="METRA",
            device=device,
            sample_cpu=self.compute.sample_cpu,
            num_train_per_epoch=self.update.num_train_per_epoch,

            sd_batch_norm=self.encoder.sd_batch_norm,
            skill_dynamics_obs_dim=obs_dim,
            trans_minibatch_size=self.update.trans_minibatch_size,
            trans_optimization_epochs=self.update.trans_optimization_epochs,

            discount=self.sac.discount,
            discrete=self.skill.discrete,
            unit_length=self.skill.unit_length,
            batch_size=self.rollout.traj_batch_size,

            snapshot_dir=str(work_dir),
            use_encoder=self.encoder.use_encoder,
            encoder_type=self.encoder.encoder_type,
            finetune_encoder=self.encoder.finetune_encoder,
            spectral_normalization=self.encoder.spectral_normalization,
            model_master_nonlinearity=self.encoder.model_master_nonlinearity,
            model_master_dim=self.encoder.model_master_dim,
            model_master_num_layers=self.encoder.model_master_num_layers,
            lr_op=self.encoder.lr_op,
            lr_te=self.encoder.lr_te,

            dual_lr=self.dual.dual_lr,
            sac_lr_q=self.sac.sac_lr_q,
            sac_lr_a=self.sac.sac_lr_a,
            seed=self.seed,

            use_target_traj_encoder=self.novelty.use_target_traj_encoder,
            grad_clip_norm=self.policy.grad_clip_norm,
            actor_init_std=self.policy.actor_init_std,
            actor_max_log_std=self.policy.actor_max_log_std,

            use_kme=self.novelty.use_kme,
            update_idk=self.novelty.update_idk,
            idk_subsample_size=self.novelty.idk_subsample_size,
            idk_init=self.novelty.idk_init,
            idk_from=self.novelty.idk_from,
            idk_groups=self.novelty.idk_groups,
            kernel_map=self.novelty.kernel_map,
            use_novelty_reward=self.novelty.use_novelty_reward,

            stage=self.skill.stage,
            skill_policy_path=self.skill.skill_policy_path,

            policy_delay=self.sac.policy_delay,
            actor_start_steps=self.sac.actor_start_steps,

            actor_critic_backbone=self.backbone.actor_critic_backbone,
            simba_actor_hidden_dim=self.backbone.simba_actor_hidden_dim,
            simba_actor_num_blocks=self.backbone.simba_actor_num_blocks,
            simba_critic_hidden_dim=self.backbone.simba_critic_hidden_dim,
            simba_critic_num_blocks=self.backbone.simba_critic_num_blocks,
            simba_mlp_ratio=self.backbone.simba_mlp_ratio,
            simba_rsnorm_momentum=self.backbone.simba_rsnorm_momentum,
            simba_rsnorm_eps=self.backbone.simba_rsnorm_eps,
            simba_ln_eps=self.backbone.simba_ln_eps,
        )


def build_env(args: argparse.Namespace):
    return make_env(mode="train", config=args)

def build_replay_buffer(env, cfg: ExperimentConfig) -> PathBufferEx:
    pixel_shape = env.spec.observation_space.shape if cfg.encoder.use_encoder else None
    return PathBufferEx(
        capacity_in_transitions=int(cfg.sac.max_buffer_size),
        pixel_shape=pixel_shape,
    )

def build_agent(cfg: ExperimentConfig, *, env, replay_buffer, work_dir: Path, device: torch.device):
    kwargs = cfg.to_agent_kwargs(env=env, replay_buffer=replay_buffer, work_dir=work_dir, device=device)
    agent = metra.DRQ_METRAAgent(**kwargs)
    agent.setup_logger(str(work_dir))
    return agent


class Workspace:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = ExperimentConfig.from_args(args)

        self.work_dir = self.cfg.run_dir.build(args)

        utils.set_seed_everywhere(self.cfg.seed)
        self.device = self.cfg.compute.device()
        self.env = build_env(args)
        self.replay_buffer = build_replay_buffer(self.env, self.cfg)
        self.agent = build_agent(self.cfg, env=self.env, replay_buffer=self.replay_buffer, work_dir=self.work_dir, device=self.device)

    def run(self):
        self.agent.train(n_epochs=self.cfg.schedule.n_epochs)
        self.agent.save(epoch="final", pt_save=True)



def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log_dir", type=str, default=os.environ.get("METRA_LOG_DIR", "/share/shangyy"))
    return parser


def main():
    args = get_argparser().parse_args()
    ws = Workspace(args)
    ws.run()

if __name__ == "__main__":
    main()
