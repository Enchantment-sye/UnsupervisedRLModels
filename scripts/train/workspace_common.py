import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from envs import make_env
from memory.replay_buffer import PathBufferEx
from utils.checkpointing import infer_run_dir_from_artifact, infer_run_dir_from_checkpoint, resolve_resume_path
from core.stage_contract import get_base_algo_name


def configure_runtime():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="ing")
    _ensure_mujoco_runtime_path()
    torch.backends.cudnn.benchmark = True


def _startup_is_quiet():
    return os.environ.get("METRA_STARTUP_QUIET", "0").strip().lower() in ("1", "true", "yes", "on")


def _startup_print(message):
    if not _startup_is_quiet():
        print(message)


def _ensure_mujoco_runtime_path():
    candidate_paths = [
        os.path.expanduser("~/.mujoco/mujoco210/bin"),
        "/usr/lib/nvidia",
    ]

    existing = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    changed = False
    for candidate in candidate_paths:
        if not os.path.isdir(candidate) or candidate in existing:
            continue
        existing.append(candidate)
        changed = True

    if changed:
        os.environ["LD_LIBRARY_PATH"] = ":".join(existing)


@dataclass
class WorkspaceContext:
    work_dir: str
    device: torch.device
    resume_checkpoint: Optional[str] = None
    is_resume: bool = False
    source_run_dir: Optional[str] = None

    @classmethod
    def create(cls, cfg, resume_from=None):
        device = _resolve_device(cfg)
        if resume_from:
            checkpoint_path = resolve_resume_path(resume_from)
            work_dir = infer_run_dir_from_checkpoint(checkpoint_path)
            if not os.path.isdir(work_dir):
                raise FileNotFoundError(f"Resolved resume work_dir does not exist: {work_dir}")
            if not os.access(work_dir, os.W_OK):
                raise PermissionError(f"Resume work_dir is not writable: {work_dir}")
            os.makedirs(os.path.join(work_dir, "models"), exist_ok=True)
            _startup_print(f"Resuming workspace directory: {work_dir}")
            return cls(
                work_dir=work_dir,
                device=device,
                resume_checkpoint=checkpoint_path,
                is_resume=True,
                source_run_dir=work_dir,
            )

        work_dir = _build_train_work_dir(cfg)
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, "models"), exist_ok=True)
        _startup_print(f"Workspace directory: {work_dir}")
        return cls(work_dir=work_dir, device=device, source_run_dir=work_dir)

    @classmethod
    def create_eval(cls, cfg, *, eval_mode="standard", resume_from=None, source_artifact=None):
        device = _resolve_device(cfg)
        checkpoint_path = None
        source_run_dir = None

        if resume_from:
            checkpoint_path = resolve_resume_path(resume_from)
            source_run_dir = infer_run_dir_from_checkpoint(checkpoint_path)
        elif source_artifact:
            source_run_dir = infer_run_dir_from_artifact(source_artifact)

        eval_root = _build_eval_root(cfg, eval_mode, source_run_dir=source_run_dir)
        timestamp = __import__("time").strftime("%Y%m%d-%H%M%S")
        work_dir = os.path.join(eval_root, f"{timestamp}_seed{cfg.seed}")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, "models"), exist_ok=True)
        _startup_print(f"Eval directory: {work_dir}")
        return cls(
            work_dir=work_dir,
            device=device,
            resume_checkpoint=checkpoint_path,
            is_resume=bool(resume_from),
            source_run_dir=source_run_dir,
        )


def save_args_json(work_dir, args, filename="args.json"):
    with open(os.path.join(work_dir, filename), "w") as fh:
        json.dump(vars(args), fh, sort_keys=True, indent=4)


def build_env_and_replay_buffer(cfg, args):
    env = make_env(mode="train", config=args)
    pixel_shape = env.spec.observation_space.shape if cfg.net.encoder else None
    replay_buffer = PathBufferEx(
        capacity_in_transitions=int(cfg.train.sac_max_buffer_size),
        pixel_shape=pixel_shape,
    )
    return env, replay_buffer


def _resolve_device(cfg):
    return torch.device("cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu")


def _build_train_work_dir(cfg):
    base_dir, manual_workspace_root = _resolve_workspace_base_dir(cfg)
    timestamp = __import__("time").strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"{timestamp}_seed{cfg.seed}"

    if manual_workspace_root:
        return os.path.join(base_dir, run_dir_name)

    stage_dir = cfg.log.stage if cfg.log.stage else "unknown_stage"
    algo_dir = cfg.algo.algo
    encoder_stage = "finetune_visual" if cfg.net.finetune_encoder else "freeze_visual"

    path_components = [base_dir, cfg.env.task, algo_dir, stage_dir, encoder_stage]
    if get_base_algo_name(cfg) == "iksd":
        path_components.append(str(cfg.algo.idk_subsample_size))
    path_components.append(run_dir_name)
    return os.path.join(*path_components)


def _build_eval_root(cfg, eval_mode: str, source_run_dir: Optional[str] = None) -> str:
    if source_run_dir:
        source_eval_root = os.path.join(source_run_dir, "evals", eval_mode)
        if _path_is_writable_or_creatable(source_eval_root):
            return source_eval_root

        fallback_base_dir, _ = _resolve_workspace_base_dir(cfg)
        stage_dir = cfg.log.stage if cfg.log.stage else "unknown_stage"
        encoder_stage = "finetune_visual" if cfg.net.finetune_encoder else "freeze_visual"
        fallback_eval_root = os.path.join(
            fallback_base_dir,
            cfg.env.task,
            cfg.algo.algo,
            stage_dir,
            encoder_stage,
            "evals",
            eval_mode,
        )
        _startup_print(
            f"Warning: source eval root {source_eval_root} is not writable, using {fallback_eval_root}"
        )
        return fallback_eval_root

    base_dir, _ = _resolve_workspace_base_dir(cfg)
    stage_dir = cfg.log.stage if cfg.log.stage else "unknown_stage"
    encoder_stage = "finetune_visual" if cfg.net.finetune_encoder else "freeze_visual"
    return os.path.join(base_dir, cfg.env.task, cfg.algo.algo, stage_dir, encoder_stage, "evals", eval_mode)


def _resolve_workspace_base_dir(cfg):
    default_workspace_root = "/share/shangyy"
    default_fallback_root = "/share/shangyy"
    env_workspace_root = os.environ.get("METRA_WORK_ROOT")
    cfg_workspace_root = cfg.log.workspace_root
    manual_workspace_root = env_workspace_root
    if manual_workspace_root is None and cfg_workspace_root not in (None, "", default_workspace_root):
        manual_workspace_root = cfg_workspace_root
    configured_root = manual_workspace_root or cfg_workspace_root or default_workspace_root
    base_dir = os.path.abspath(os.path.expanduser(configured_root))

    if os.path.exists(base_dir):
        if not os.access(base_dir, os.W_OK):
            fallback_dir = _resolve_workspace_fallback(default_fallback_root)
            _startup_print(f"Warning: workspace root {base_dir} not writable, using {fallback_dir}")
            base_dir = fallback_dir
    else:
        parent_dir = os.path.dirname(base_dir) or "."
        if not os.path.exists(parent_dir) or not os.access(parent_dir, os.W_OK):
            fallback_dir = _resolve_workspace_fallback(default_fallback_root)
            _startup_print(f"Warning: cannot create workspace root {base_dir}, using {fallback_dir}")
            base_dir = fallback_dir

    os.makedirs(base_dir, exist_ok=True)
    return base_dir, manual_workspace_root


def _resolve_workspace_fallback(preferred_root: str) -> str:
    preferred_root = os.path.abspath(os.path.expanduser(preferred_root))
    if os.path.exists(preferred_root):
        if os.access(preferred_root, os.W_OK):
            return preferred_root
    else:
        parent_dir = os.path.dirname(preferred_root) or "."
        if os.path.exists(parent_dir) and os.access(parent_dir, os.W_OK):
            return preferred_root

    import tempfile

    return tempfile.gettempdir()


def _path_is_writable_or_creatable(path: str) -> bool:
    target = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(target):
        return os.path.isdir(target) and os.access(target, os.W_OK)

    probe = target
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent

    return os.path.isdir(probe) and os.access(probe, os.W_OK)
