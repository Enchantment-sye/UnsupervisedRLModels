from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


class RunningMeanStd:
    def __init__(self, device="cpu", eps: float = 1e-8):
        self.device = torch.device(device)
        self.eps = float(eps)
        self.count = torch.zeros((), device=self.device)
        self.mean = torch.zeros((), device=self.device)
        self.m2 = torch.zeros((), device=self.device)

    @property
    def var(self):
        return self.m2 / self.count.clamp_min(1.0)

    @property
    def std(self):
        return torch.sqrt(self.var + self.eps)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        x = values.detach().to(self.device).float().reshape(-1)
        if x.numel() == 0:
            return
        batch_count = torch.as_tensor(float(x.numel()), device=self.device)
        batch_mean = x.mean()
        batch_m2 = ((x - batch_mean) ** 2).sum()
        if float(self.count.item()) == 0.0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        self.m2 = self.m2 + batch_m2 + delta.square() * self.count * batch_count / total
        self.count = total

    @torch.no_grad()
    def normalize_clip(self, values: torch.Tensor, clip: float, *, update: bool = True) -> torch.Tensor:
        x = values.detach().to(self.device).float()
        if update:
            self.update(x)
        return ((x - self.mean) / self.std).clamp(-float(clip), float(clip))


class _ZBuffer:
    def __init__(self, max_size: int, z_dim: int, device, *, reservoir: bool):
        self.max_size = int(max_size)
        self.z_dim = int(z_dim)
        self.device = torch.device(device)
        self.reservoir = bool(reservoir)
        self.data = torch.empty(self.max_size, self.z_dim, device=self.device)
        self.size = 0
        self.next_idx = 0
        self.total_seen = 0

    def tensor(self) -> torch.Tensor:
        return self.data[: self.size]

    @torch.no_grad()
    def set_data(self, z: torch.Tensor, *, keep_last: bool) -> None:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        original_n = int(z.shape[0])
        if z.shape[0] > self.max_size:
            if keep_last:
                z = z[-self.max_size :]
            else:
                idx = torch.randperm(z.shape[0], device=z.device)[: self.max_size]
                z = z[idx]
        self.size = int(z.shape[0])
        if self.size:
            self.data[: self.size].copy_(z)
        self.next_idx = self.size % self.max_size
        self.total_seen = original_n

    @torch.no_grad()
    def add_batch(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        added = []
        removed = []
        for row in z:
            add, rem = self._add_one(row)
            if add is not None:
                added.append(add)
            if rem is not None:
                removed.append(rem)
        return self._stack_or_empty(added), self._stack_or_empty(removed)

    def _add_one(self, row: torch.Tensor):
        self.total_seen += 1
        if self.reservoir:
            if self.size < self.max_size:
                idx = self.size
                self.size += 1
                removed = None
            else:
                idx = random.randrange(self.total_seen)
                if idx >= self.max_size:
                    return None, None
                removed = self.data[idx].clone()
            self.data[idx].copy_(row)
            return row.clone(), removed

        if self.size < self.max_size:
            idx = self.size
            self.size += 1
            removed = None
        else:
            idx = self.next_idx
            removed = self.data[idx].clone()
            self.next_idx = (self.next_idx + 1) % self.max_size
        self.data[idx].copy_(row)
        return row.clone(), removed

    def _stack_or_empty(self, rows):
        if rows:
            return torch.stack(rows, dim=0).to(self.device)
        return torch.empty(0, self.z_dim, device=self.device)


class _NNMassModel:
    def __init__(self, *, c: int, psi: int, alpha: float, z_dim: int, device):
        self.c = int(c)
        self.psi = int(psi)
        self.alpha = float(alpha)
        self.z_dim = int(z_dim)
        self.device = torch.device(device)
        self.anchors: Optional[torch.Tensor] = None
        self.counts: Optional[torch.Tensor] = None
        self.n = 0

    @property
    def built(self) -> bool:
        return self.anchors is not None and self.counts is not None and self.n > 0

    @torch.no_grad()
    def build(self, z: torch.Tensor) -> None:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        self.n = int(z.shape[0])
        if self.n <= 0:
            self.anchors = None
            self.counts = None
            return
        idx = torch.randint(self.n, (self.c, self.psi), device=self.device)
        self.anchors = z[idx].clone()
        self._recompute_all_counts(z)

    @torch.no_grad()
    def refresh_members(self, z: torch.Tensor, refresh_num: int) -> None:
        if not self.built:
            self.build(z)
            return
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        self.n = int(z.shape[0])
        if self.n <= 0:
            self.anchors = None
            self.counts = None
            return
        k = max(1, min(int(refresh_num), self.c))
        members = torch.randperm(self.c, device=self.device)[:k]
        idx = torch.randint(self.n, (k, self.psi), device=self.device)
        self.anchors[members] = z[idx]
        labels = self.assign(z)
        for member in members.tolist():
            self.counts[member].zero_()
            self.counts[member].scatter_add_(
                0,
                labels[:, member],
                torch.ones(labels.shape[0], device=self.device, dtype=self.counts.dtype),
            )

    @torch.no_grad()
    def _recompute_all_counts(self, z: torch.Tensor) -> None:
        labels = self.assign(z)
        self.counts = torch.zeros(self.c, self.psi, device=self.device)
        ones = torch.ones(labels.shape[0], device=self.device)
        for member in range(self.c):
            self.counts[member].scatter_add_(0, labels[:, member], ones)

    @torch.no_grad()
    def assign(self, z: torch.Tensor) -> torch.Tensor:
        if self.anchors is None:
            raise RuntimeError("NN-MASS model has not been built")
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        labels = []
        for member in range(self.c):
            dist = (z[:, None, :] - self.anchors[member][None, :, :]).square().sum(dim=-1)
            labels.append(dist.argmin(dim=1))
        return torch.stack(labels, dim=1)

    @torch.no_grad()
    def update_counts(self, z: torch.Tensor, delta: float) -> None:
        if not self.built:
            return
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        if z.numel() == 0:
            return
        labels = self.assign(z)
        vals = torch.full((z.shape[0],), float(delta), device=self.device)
        for member in range(self.c):
            self.counts[member].scatter_add_(0, labels[:, member], vals)
        self.counts.clamp_(min=0.0)
        self.n = max(0, int(round(float(self.counts[0].sum().item()))))

    @torch.no_grad()
    def reward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        if not self.built:
            return torch.zeros(z.shape[0], device=self.device)
        labels = self.assign(z)
        rewards = []
        denom = float(self.n) + self.alpha * float(self.psi)
        for member in range(self.c):
            cnt = self.counts[member].gather(0, labels[:, member])
            rewards.append(torch.log(z.new_tensor(denom) / (cnt + self.alpha)))
        return torch.stack(rewards, dim=0).mean(dim=0)

    @torch.no_grad()
    def stats(self, z: torch.Tensor) -> Dict[str, float]:
        if not self.built:
            return {
                "empty_cell_ratio": 1.0,
                "cell_entropy": 0.0,
                "mean_cell_count": 0.0,
                "mean_anchor_distance": 0.0,
            }
        counts = self.counts
        empty = (counts <= 0).float().mean().item()
        probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean().item()
        mean_count = counts.mean().item()
        mean_dist = self._mean_anchor_distance(z)
        return {
            "empty_cell_ratio": float(empty),
            "cell_entropy": float(entropy),
            "mean_cell_count": float(mean_count),
            "mean_anchor_distance": float(mean_dist),
        }

    @torch.no_grad()
    def _mean_anchor_distance(self, z: torch.Tensor) -> float:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        if z.numel() == 0 or not self.built:
            return 0.0
        if z.shape[0] > 4096:
            z = z[torch.randperm(z.shape[0], device=self.device)[:4096]]
        dists = []
        for member in range(self.c):
            dist = (z[:, None, :] - self.anchors[member][None, :, :]).square().sum(dim=-1)
            dists.append(torch.sqrt(dist.min(dim=1).values.clamp_min(0.0)))
        return torch.stack(dists, dim=0).mean().item()


@dataclass
class MassRewardComponents:
    short: torch.Tensor
    long: torch.Tensor
    total: torch.Tensor


class StreamingNNMass:
    def __init__(
        self,
        *,
        z_dim: int,
        c: int = 64,
        psi: int = 128,
        alpha: float = 1.0,
        short_size: int = 50000,
        long_size: int = 300000,
        w_short: float = 0.7,
        w_long: float = 0.3,
        reward_clip: float = 5.0,
        device="cpu",
    ):
        self.z_dim = int(z_dim)
        self.device = torch.device(device)
        self.c = int(c)
        self.psi = int(psi)
        self.alpha = float(alpha)
        self.w_short = float(w_short)
        self.w_long = float(w_long)
        self.reward_clip = float(reward_clip)
        self.short_buffer = _ZBuffer(short_size, self.z_dim, self.device, reservoir=False)
        self.long_buffer = _ZBuffer(long_size, self.z_dim, self.device, reservoir=True)
        self.short_model = _NNMassModel(c=self.c, psi=self.psi, alpha=self.alpha, z_dim=self.z_dim, device=self.device)
        self.long_model = _NNMassModel(c=self.c, psi=self.psi, alpha=self.alpha, z_dim=self.z_dim, device=self.device)
        self.rms = RunningMeanStd(device=self.device)
        self.repartition_count = 0
        self.refresh_count = 0

    @property
    def short_size(self) -> int:
        return self.short_buffer.size

    @property
    def long_size(self) -> int:
        return self.long_buffer.size

    @torch.no_grad()
    def build_initial_partitions(self, z: torch.Tensor) -> Dict[str, float]:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        self.short_buffer.set_data(z, keep_last=True)
        self.long_buffer.set_data(z, keep_last=False)
        self.short_model.build(self.short_buffer.tensor())
        self.long_model.build(self.long_buffer.tensor())
        self.repartition_count += 1
        return self.stats()

    @torch.no_grad()
    def add_z(self, z: torch.Tensor) -> None:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        short_added, short_removed = self.short_buffer.add_batch(z)
        long_added, long_removed = self.long_buffer.add_batch(z)
        self.short_model.update_counts(short_removed, -1.0)
        self.short_model.update_counts(short_added, 1.0)
        self.long_model.update_counts(long_removed, -1.0)
        self.long_model.update_counts(long_added, 1.0)

    @torch.no_grad()
    def repartition(self) -> Dict[str, float]:
        self.short_model.build(self.short_buffer.tensor())
        self.long_model.build(self.long_buffer.tensor())
        self.repartition_count += 1
        return self.stats()

    @torch.no_grad()
    def rolling_refresh(self, refresh_num: int) -> Dict[str, float]:
        self.short_model.refresh_members(self.short_buffer.tensor(), refresh_num)
        self.long_model.refresh_members(self.long_buffer.tensor(), refresh_num)
        self.refresh_count += 1
        return self.stats()

    @torch.no_grad()
    def reward_components(self, z: torch.Tensor) -> MassRewardComponents:
        z = z.detach().to(self.device).float().reshape(-1, self.z_dim)
        short_reward = self.short_model.reward(z)
        long_reward = self.long_model.reward(z)
        total = self.w_short * short_reward + self.w_long * long_reward
        return MassRewardComponents(short=short_reward, long=long_reward, total=total)

    @torch.no_grad()
    def reward_batch(self, z: torch.Tensor) -> torch.Tensor:
        return self.reward_components(z).total

    @torch.no_grad()
    def normalize_clip(self, rewards: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        return self.rms.normalize_clip(rewards, self.reward_clip, update=update)

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        short_stats = self.short_model.stats(self.short_buffer.tensor())
        long_stats = self.long_model.stats(self.long_buffer.tensor())
        return {
            "short_size": float(self.short_buffer.size),
            "long_size": float(self.long_buffer.size),
            "empty_cell_ratio_short": short_stats["empty_cell_ratio"],
            "empty_cell_ratio_long": long_stats["empty_cell_ratio"],
            "cell_entropy_short": short_stats["cell_entropy"],
            "cell_entropy_long": long_stats["cell_entropy"],
            "mean_cell_count_short": short_stats["mean_cell_count"],
            "mean_cell_count_long": long_stats["mean_cell_count"],
            "mean_anchor_distance_short": short_stats["mean_anchor_distance"],
            "mean_anchor_distance_long": long_stats["mean_anchor_distance"],
            "repartition_count": float(self.repartition_count),
            "refresh_count": float(self.refresh_count),
        }
