from __future__ import annotations

from typing import Dict, Optional

import torch

from .nn_mass import StreamingNNMass


class MassRewardAdapter:
    """External reward replacement layer for existing actor-critic updates."""

    def __init__(
        self,
        *,
        coverage_encoder,
        mass_model: StreamingNNMass,
        lambda_action: float = 1e-3,
        lambda_delta_action: float = 1e-3,
        lambda_done: float = 5.0,
        encode_batch_size: int = 128,
        device="cpu",
    ):
        self.coverage_encoder = coverage_encoder
        self.mass_model = mass_model
        self.lambda_action = float(lambda_action)
        self.lambda_delta_action = float(lambda_delta_action)
        self.lambda_done = float(lambda_done)
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.device = torch.device(device)

    @torch.no_grad()
    def compute_batch_reward(self, batch: Dict[str, torch.Tensor], *, update_rms: bool = False) -> Dict[str, torch.Tensor]:
        z_next = self.encode_observations(batch["next_obs"])
        return self.compute_from_z(
            z_next,
            actions=batch["actions"],
            prev_actions=batch.get("prev_actions"),
            dones=batch.get("dones"),
            update_rms=update_rms,
        )

    @torch.no_grad()
    def compute_step_reward(
        self,
        next_obs,
        *,
        action,
        prev_action,
        done,
        update_rms: bool = True,
    ) -> Dict[str, torch.Tensor]:
        next_obs_t = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)
        z_next = self.coverage_encoder(next_obs_t).detach()
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(1, -1)
        prev_t = torch.as_tensor(prev_action, device=self.device, dtype=torch.float32).reshape(1, -1)
        done_t = torch.as_tensor([float(done)], device=self.device, dtype=torch.float32)
        out = self.compute_from_z(z_next, actions=action_t, prev_actions=prev_t, dones=done_t, update_rms=update_rms)
        out["z_next"] = z_next.detach().to(self.mass_model.device).float()
        return out

    @torch.no_grad()
    def encode_observations(self, obs) -> torch.Tensor:
        if torch.is_tensor(obs):
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            total = int(obs.shape[0])
        else:
            if len(obs) == 0:
                return torch.empty(0, 0, device=self.mass_model.device)
            total = int(obs.shape[0])
        chunks = []
        for start in range(0, total, self.encode_batch_size):
            end = min(start + self.encode_batch_size, total)
            obs_chunk = obs[start:end]
            if torch.is_tensor(obs_chunk):
                obs_chunk = obs_chunk.to(self.device).float()
            else:
                obs_chunk = torch.as_tensor(obs_chunk, device=self.device, dtype=torch.float32)
            z_chunk = self.coverage_encoder(obs_chunk).detach().to(self.mass_model.device).float()
            chunks.append(z_chunk)
        if not chunks:
            return torch.empty(0, 0, device=self.mass_model.device)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def compute_from_z(
        self,
        z_next: torch.Tensor,
        *,
        actions,
        prev_actions: Optional[torch.Tensor],
        dones: Optional[torch.Tensor],
        update_rms: bool,
    ) -> Dict[str, torch.Tensor]:
        z_next = z_next.detach().to(self.mass_model.device).float()
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).reshape(z_next.shape[0], -1)
        if prev_actions is None:
            prev_actions = torch.zeros_like(actions)
        else:
            prev_actions = torch.as_tensor(prev_actions, device=self.device, dtype=torch.float32).reshape_as(actions)
        if dones is None:
            dones = torch.zeros(z_next.shape[0], device=self.device)
        else:
            dones = torch.as_tensor(dones, device=self.device, dtype=torch.float32).reshape(z_next.shape[0])

        components = self.mass_model.reward_components(z_next)
        r_cov_mass = components.total.detach()
        r_norm = self.mass_model.normalize_clip(r_cov_mass, update=update_rms).to(self.device)
        r_cov = r_cov_mass.to(self.device)

        action_penalty = self.lambda_action * actions.square().sum(dim=-1)
        delta_action = actions - prev_actions
        delta_penalty = self.lambda_delta_action * delta_action.square().sum(dim=-1)
        done_penalty = self.lambda_done * dones
        r_int = r_norm - action_penalty - delta_penalty - done_penalty

        return {
            "rewards": r_int.detach(),
            "r_int": r_int.detach(),
            "r_cov": r_cov.detach(),
            "r_cov_short": components.short.detach().to(self.device),
            "r_cov_long": components.long.detach().to(self.device),
            "r_norm": r_norm.detach(),
            "action_norm": torch.linalg.vector_norm(actions, dim=-1).detach(),
            "delta_action_norm": torch.linalg.vector_norm(delta_action, dim=-1).detach(),
            "terminal_rate": dones.detach(),
        }

    @staticmethod
    def scalar_stats(prefix: str, values: torch.Tensor) -> Dict[str, float]:
        v = values.detach().float().reshape(-1)
        return {
            f"{prefix}_mean": v.mean().item(),
            f"{prefix}_std": v.std(unbiased=False).item() if v.numel() > 1 else 0.0,
            f"{prefix}_min": v.min().item(),
            f"{prefix}_max": v.max().item(),
        }
