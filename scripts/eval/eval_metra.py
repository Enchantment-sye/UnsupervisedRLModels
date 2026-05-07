import json
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TRAIN_SCRIPT_DIR = os.path.join(REPO_ROOT, 'scripts', 'train')
SRC_DIR = os.path.join(REPO_ROOT, 'src')
for path in (REPO_ROOT, TRAIN_SCRIPT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import utils, video_motion
from core.metra_agent import MetraAgent
from core.metra_config import get_parser, make_config_from_args
from core.metra_trainer import MetraTrainer
from core.metra_viz import plot_trajectories
from workspace_common import WorkspaceContext, build_env_and_replay_buffer, configure_runtime, save_args_json
from utils.checkpointing import infer_run_dir_from_artifact

configure_runtime()


EVAL_ARG_LOGGER = logging.getLogger('EvalArgs')
EVAL_SAFE_OVERRIDE_DESTS = {
    'resume_from',
    'ckpt_path',
    'skill_policy_path',
    'traj_encoder_path',
    'eval_mode',
    'seed',
    'num_random_trajectories',
    'num_video_repeats',
    'eval_record_video',
    'video_skip_frames',
    'eval_plot_axis',
    'ikse',
    'metric_num_sampled_points',
    'dbi_num_rollouts_per_skill',
    'temporal_graph_num_warmup_rollouts',
    'temporal_graph_rollouts_per_skill',
    'temporal_graph_knn_k',
    'temporal_bridge_cost',
    'soft_dtw_gamma',
    'motion_analysis_enabled',
    'motion_analysis_video_path',
    'motion_analysis_resize_h',
    'motion_analysis_resize_w',
    'motion_analysis_blur_kernel',
    'motion_analysis_frame_gap',
    'motion_analysis_pixel_threshold_mode',
    'motion_analysis_fixed_tau_p',
    'motion_analysis_smooth_window',
    'motion_analysis_large_motion_threshold',
    'motion_analysis_eps',
}
STRUCTURE_KEY_DESTS = {
    'task',
    'algo',
    'stage',
    'dim_skill',
    'discrete',
    'encoder',
    'use_cascade',
    'num_policy_levels',
    'ac_backbone',
    'cascade_init_from_prev',
    'cascade_gate_type',
    'cascade_min_lambda',
    'cascade_max_lambda',
    'use_hierarchical_policy',
    'use_hierarchical_skill',
    'num_skill_levels',
    'use_hierarchical_phi',
    'hierarchical_phi_depth',
    'use_kme',
    'idk_subsample_size',
    'action_repeat',
    'time_limit',
    'render_size',
    'flatten_obs',
    'camera',
    'dmc_camera',
    'unit_length',
    'normalizer_type',
    'traj_latent_norm',
    'traj_latent_norm_eps',
    'dual_dist',
    'encoder_type',
    'finetune_encoder',
}


class EvalWorkspace:
    """Lightweight eval workspace that mirrors the training construction path."""

    def __init__(self, args):
        self.args = args if getattr(args, '_eval_args_normalized', False) else normalize_eval_args(args)
        self.cfg = make_config_from_args(self.args)
        self.ctx = WorkspaceContext.create_eval(
            self.cfg,
            eval_mode=self.args.eval_mode,
            resume_from=getattr(self.args, 'resume_from', None),
            source_artifact=_resolve_source_artifact(self.args),
        )
        save_args_json(self.ctx.work_dir, self.args, filename='eval_args.json')

        utils.set_seed_everywhere(self.cfg.seed)
        self.env, self.replay_buffer = build_env_and_replay_buffer(self.cfg, self.args)
        self.agent = MetraAgent(self.cfg, self.env, self.replay_buffer)
        self.trainer = MetraTrainer(self.cfg, self.agent, self.env, self.replay_buffer, self.ctx.work_dir)

        # Optional compatibility attributes for shared visualization helpers.
        self.agent.snapshot_dir = self.ctx.work_dir
        self.agent.writer = self.trainer.writer
        self.agent.logger = self.trainer.logger

        self._load_checkpoints()

    def _load_checkpoints(self):
        if self.ctx.resume_checkpoint:
            self.trainer.load_resume_checkpoint(self.ctx.resume_checkpoint)
        else:
            self.agent.load_component_checkpoints(
                self.args.skill_policy_path,
                traj_encoder_path=self.args.traj_encoder_path,
            )

        self.agent.step_itr = self.trainer.step_itr

        if self.args.eval_mode == 'trajectories' and self.agent.traj_encoder is None:
            raise ValueError(
                'Trajectory analysis requires a trajectory encoder. '
                'Use stage=pre_training with --traj-encoder-path or --resume_from a pretraining checkpoint.'
            )

    def run(self):
        self.trainer._set_models_mode('eval')
        if self.agent.traj_encoder is not None:
            self.agent.traj_encoder.eval()
        if self.agent.target_traj_encoder is not None:
            self.agent.target_traj_encoder.eval()

        self.trainer.task_adapter.on_train_start()

        if self.args.eval_mode == 'trajectories':
            plot_trajectories(
                self.agent,
                self.trainer.task_adapter,
                snapshot_dir=self.ctx.work_dir,
                writer=self.trainer.writer,
                logger=self.trainer.logger,
                step_itr=self.trainer.step_itr,
                rollout_seed=self.cfg.seed + 100 + self.trainer.total_epoch,
            )
            policy_coverage_trajectories = self.trainer.task_adapter.collect_policy_coverage_trajectories(
                self.trainer.total_epoch,
            )
            policy_coverage_metrics = self.trainer.task_adapter.compute_policy_coverage_metrics(
                policy_coverage_trajectories,
            )
            self.trainer.task_adapter.log_policy_coverage_metrics_to_logger(
                policy_coverage_metrics,
                self.trainer.step_itr,
                print_to_stdout=True,
            )
        else:
            eval_result = self.trainer.task_adapter.evaluate(
                self.trainer.step_itr,
                self.trainer.total_epoch,
                self.trainer.writer,
                log_policy_coverage_to_writer=False,
            )
            self.trainer.task_adapter.log_policy_coverage_metrics_to_logger(
                eval_result.get('policy_coverage_metrics', {}),
                self.trainer.step_itr,
                print_to_stdout=True,
            )
            self.trainer.log_diagnostics()

        self.trainer.writer.flush()


def get_eval_parser():
    parser = get_parser()
    group = parser.add_argument_group('Evaluation')
    group.add_argument('--ckpt-path', dest='ckpt_path', type=str, default=None)
    group.add_argument('--traj-encoder-path', dest='traj_encoder_path', type=str, default=None)
    group.add_argument('--eval-mode', dest='eval_mode', type=str, default='standard', choices=['standard', 'trajectories'])
    return parser


def _compute_cli_override_dests(parser, argv):
    if parser is None or argv is None:
        return set()

    option_to_dest = {}
    for action in parser._actions:
        for option_string in getattr(action, 'option_strings', ()):
            option_to_dest[option_string] = action.dest

    alias_override_dests = {
        'ckpt_path': {'skill_policy_path'},
    }

    cli_overrides = set()
    for token in argv:
        if not token.startswith('-') or token == '-':
            continue
        option = token.split('=', 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            cli_overrides.add(dest)
            cli_overrides.update(alias_override_dests.get(dest, ()))
    return cli_overrides


def _load_source_run_args(args):
    if getattr(args, 'resume_from', None):
        return None, None

    source_artifact = _resolve_source_artifact(args)
    if not source_artifact:
        return None, None

    try:
        run_dir = infer_run_dir_from_artifact(source_artifact)
    except FileNotFoundError:
        return None, None

    for candidate_name in ('args.json', 'resume_args.json'):
        candidate_path = os.path.join(run_dir, candidate_name)
        if not os.path.isfile(candidate_path):
            continue
        with open(candidate_path, 'r') as fh:
            return json.load(fh), candidate_path

    return None, None


def _hydrate_component_eval_args(args, *, parser=None, argv=None):
    source_args, source_args_path = _load_source_run_args(args)
    if not source_args:
        return args

    cli_overrides = _compute_cli_override_dests(parser, argv)
    overwritten_structure_keys = []
    for key, source_value in source_args.items():
        if not hasattr(args, key):
            continue

        current_value = getattr(args, key)
        if key in EVAL_SAFE_OVERRIDE_DESTS and key in cli_overrides:
            continue

        if key in STRUCTURE_KEY_DESTS and key in cli_overrides and current_value != source_value:
            overwritten_structure_keys.append((key, current_value, source_value))

        setattr(args, key, source_value)

    if overwritten_structure_keys:
        details = ", ".join(
            f"{key}: cli={current!r} -> source={source!r}"
            for key, current, source in overwritten_structure_keys
        )
        EVAL_ARG_LOGGER.warning(
            "Component eval restored training structure from %s and overrode conflicting CLI values: %s",
            source_args_path,
            details,
        )

    return args




def _has_motion_analysis_video_path(args):
    return bool(getattr(args, 'motion_analysis_video_path', ''))


def _normalize_motion_analysis_args(args):
    video_path = getattr(args, 'motion_analysis_video_path', None)
    if video_path:
        args.motion_analysis_video_path = _expand_path(video_path)
        if not os.path.isfile(args.motion_analysis_video_path):
            raise FileNotFoundError(f'motion analysis video not found: {args.motion_analysis_video_path}')
        args.motion_analysis_enabled = 1
    return args


def _build_motion_analysis_logger():
    logger = logging.getLogger('MotionAnalysisEval')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger


def run_motion_analysis_from_file(args):
    cfg = make_config_from_args(args)
    logger = _build_motion_analysis_logger()
    result = video_motion.analyze_montage_video(
        args.motion_analysis_video_path,
        discrete=args.discrete,
        dim_skill=args.dim_skill,
        num_video_repeats=args.num_video_repeats,
        cfg=cfg.motion_analysis,
        logger=logger,
    )
    video_motion.log_motion_analysis(result, logger=logger)
    return result


def normalize_eval_args(args, *, parser=None, argv=None):
    args = _normalize_motion_analysis_args(args)

    if getattr(args, 'ckpt_path', None):
        args.skill_policy_path = args.ckpt_path

    if getattr(args, 'resume_from', None):
        args.resume_from = _expand_path(args.resume_from)

    if getattr(args, 'skill_policy_path', None):
        args.skill_policy_path = _expand_path(args.skill_policy_path)
        if not os.path.isfile(args.skill_policy_path):
            raise FileNotFoundError(f'skill policy checkpoint not found: {args.skill_policy_path}')

    if getattr(args, 'traj_encoder_path', None):
        args.traj_encoder_path = _expand_path(args.traj_encoder_path)
        if not os.path.isfile(args.traj_encoder_path):
            raise FileNotFoundError(f'trajectory encoder checkpoint not found: {args.traj_encoder_path}')
    elif getattr(args, 'skill_policy_path', None):
        inferred_traj_path = os.path.join(os.path.dirname(args.skill_policy_path), 'traj_encoder.pt')
        if os.path.isfile(inferred_traj_path):
            args.traj_encoder_path = inferred_traj_path

    if not getattr(args, 'resume_from', None) and not getattr(args, 'skill_policy_path', None):
        raise ValueError('Evaluation requires --resume_from or --ckpt-path/--skill_policy_path.')

    if args.eval_mode == 'trajectories' and not getattr(args, 'resume_from', None) and not getattr(args, 'traj_encoder_path', None):
        raise ValueError('Trajectory analysis requires --traj-encoder-path or a sibling traj_encoder.pt next to the skill policy checkpoint.')

    args = _hydrate_component_eval_args(args, parser=parser, argv=argv)
    args._eval_args_normalized = True
    return args


def _resolve_source_artifact(args):
    if getattr(args, 'resume_from', None):
        return None
    if getattr(args, 'skill_policy_path', None):
        return args.skill_policy_path
    return getattr(args, 'traj_encoder_path', None)


def _expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def main():
    parser = get_eval_parser()
    args = parser.parse_args()
    args = _normalize_motion_analysis_args(args)

    if _has_motion_analysis_video_path(args):
        run_motion_analysis_from_file(args)
        return

    args = normalize_eval_args(args, parser=parser, argv=sys.argv[1:])

    workspace = EvalWorkspace(args)
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
