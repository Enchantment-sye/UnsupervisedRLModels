import glob
import os
import re

import torch


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_resume_path(resume_from: str) -> str:
    if not resume_from:
        raise ValueError("resume_from is empty")

    path = os.path.abspath(os.path.expanduser(resume_from))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Resume path not found: {resume_from}")

    candidate_files = [
        os.path.join(path, "latest_resume.pt"),
        os.path.join(path, "models", "latest_resume.pt"),
    ]
    for candidate in candidate_files:
        if os.path.isfile(candidate):
            return candidate

    epoch_candidates = []
    epoch_candidates.extend(glob.glob(os.path.join(path, "models", "epoch-*", "resume_state.pt")))
    epoch_candidates.extend(glob.glob(os.path.join(path, "epoch-*", "resume_state.pt")))
    if epoch_candidates:
        return sorted(epoch_candidates, key=_resume_sort_key)[-1]

    raise FileNotFoundError(
        f"Could not find a resume checkpoint under {resume_from}. "
        "Expected latest_resume.pt or epoch-*/resume_state.pt."
    )


def infer_run_dir_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        work_dir = checkpoint.get("work_dir")
        if isinstance(work_dir, str) and work_dir:
            return os.path.abspath(os.path.expanduser(work_dir))

    parent = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_path) == "latest_resume.pt":
        if os.path.basename(parent) == "models":
            return os.path.dirname(parent)
        return parent
    if os.path.basename(checkpoint_path) == "resume_state.pt":
        models_dir = os.path.dirname(parent)
        if os.path.basename(models_dir) == "models":
            return os.path.dirname(models_dir)
        return os.path.dirname(parent)
    return parent


def infer_run_dir_from_artifact(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact path not found: {path}")
    if os.path.isdir(path):
        return path

    basename = os.path.basename(path)
    if basename in ("latest_resume.pt", "resume_state.pt"):
        return infer_run_dir_from_checkpoint(path)

    parent = os.path.dirname(path)
    if os.path.basename(parent) == "models":
        return os.path.dirname(parent)

    grandparent = os.path.dirname(parent)
    if os.path.basename(grandparent) == "models":
        return os.path.dirname(grandparent)

    return parent


def _resume_sort_key(path: str):
    epoch_match = re.search(r"epoch-(\d+)", path)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    return epoch, os.path.getmtime(path)
