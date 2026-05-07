import os
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import torch
from utils.checkpointing import safe_torch_load

@dataclass
class PretrainedMETRA:
    skill_policy: torch.nn.Module
    traj_encoder: Optional[torch.nn.Module]
    discrete: Optional[bool] = None
    dim_skill: Optional[int] = None

def _load_pt(path: str, device: torch.device) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return safe_torch_load(path, map_location=device)

def _unwrap_module(obj: Any, keys: Tuple[str, ...]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Accept either:
      - a torch.nn.Module directly
      - a dict that contains a module under one of `keys`
    Return (module, meta_dict).
    """
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], torch.nn.Module):
                meta = dict(obj)
                return obj[k], meta
        # Sometimes people save state_dict; we handle that elsewhere (agent-loading path).
        raise ValueError(f"Expected module under keys {keys}, but got dict keys={list(obj.keys())}")
    if isinstance(obj, torch.nn.Module):
        return obj, {}
    raise ValueError(f"Unsupported checkpoint type: {type(obj)}")

def load_pretrained_metra(
        ckpt_dir: str,
        device: torch.device,
        skill_policy_name: str = "skill_policy.pt",
        traj_encoder_name: str = "traj_encoder.pt",
        load_traj_encoder: bool = True,
        freeze: bool = True,
        eval_mode: bool = True,
) -> PretrainedMETRA:
    """
    Load low-level METRA components for downstream usage (meta-controller).
    This does NOT require constructing DRQ_METRAAgent.

    It expects files like:
      ckpt_dir/skill_policy.pt
      ckpt_dir/traj_encoder.pt (optional)

    Compatible with either:
      torch.save(policy_module, path)
    or:
      torch.save({'policy': policy_module, 'discrete':..., 'dim_skill':...}, path)
    """
    sp_path = os.path.join(ckpt_dir, skill_policy_name)
    sp_raw = _load_pt(sp_path, device=device)
    skill_policy, sp_meta = _unwrap_module(sp_raw, keys=("policy", "skill_policy", "module"))

    te = None
    te_meta: Dict[str, Any] = {}
    if load_traj_encoder:
        te_path = os.path.join(ckpt_dir, traj_encoder_name)
        if os.path.exists(te_path):
            te_raw = _load_pt(te_path, device=device)
            te, te_meta = _unwrap_module(te_raw, keys=("traj_encoder", "encoder", "module"))

    # Move to device (if loaded on CPU)
    skill_policy.to(device)
    if te is not None:
        te.to(device)

    # Set eval + freeze if desired
    if eval_mode:
        skill_policy.eval()
        if te is not None:
            te.eval()

    if freeze:
        for p in skill_policy.parameters():
            p.requires_grad_(False)
        if te is not None:
            for p in te.parameters():
                p.requires_grad_(False)

    # Extract metadata if present
    discrete = None
    dim_skill = None
    for meta in (sp_meta, te_meta):
        if discrete is None and isinstance(meta.get("discrete", None), (bool, int)):
            discrete = bool(meta["discrete"])
        if dim_skill is None and isinstance(meta.get("dim_skill", None), int):
            dim_skill = int(meta["dim_skill"])

    return PretrainedMETRA(
        skill_policy=skill_policy,
        traj_encoder=te,
        discrete=discrete,
        dim_skill=dim_skill,
    )


def load_pretrained_into_agent(
        agent,
        ckpt_dir: str,
        device: torch.device,
        skill_policy_name: str = "skill_policy.pt",
        traj_encoder_name: str = "traj_encoder.pt",
        strict: bool = True,
) -> None:
    """
    Load pretrained weights INTO an already constructed DRQ_METRAAgent instance.

    Why this variant:
      - Keeps your optimizers/replay buffer/etc. intact
      - Safer for continuing training

    It supports two checkpoint styles:
      A) saved module: torch.save(policy_module, path)
      B) saved dict with module: torch.save({'policy': policy_module, ...}, path)
    We load via state_dict() so the agent's modules keep their identity.
    """
    # --- skill policy ---
    sp_raw = _load_pt(os.path.join(ckpt_dir, skill_policy_name), device=device)
    sp_mod, sp_meta = _unwrap_module(sp_raw, keys=("policy", "skill_policy", "module"))
    agent.skill_policy.load_state_dict(sp_mod.state_dict(), strict=strict)

    # --- traj encoder (optional) ---
    te_path = os.path.join(ckpt_dir, traj_encoder_name)
    if os.path.exists(te_path):
        te_raw = _load_pt(te_path, device=device)
        te_mod, _te_meta = _unwrap_module(te_raw, keys=("traj_encoder", "encoder", "module"))
        agent.traj_encoder.load_state_dict(te_mod.state_dict(), strict=strict)

    # move and set modes
    agent.skill_policy.to(device)
    agent.traj_encoder.to(device)
    agent.skill_policy.eval()
    agent.traj_encoder.eval()
