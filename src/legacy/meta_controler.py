# drq_url/meta_controller_ppo_discrete.py
import os
import random
import imageio.v2 as imageio

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils  # drq_url/utils.py
from metra import MeasureAndAccTime
from workers.rollout import SkillRolloutWorker, TrajectoryBatch


# =========================
# Configs
# =========================

@dataclass
class MetaPPOConfig:
    lr: float = 3e-4
    clip_ratio: float = 0.2
    train_iters: int = 10
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0


@dataclass
class TrainerConfig:
    k_steps: int = 25
    gamma: float = 0.99
    max_episode_steps: Optional[int] = None  # None -> rely on env termination
    rollout_macro_steps: int = 4096          # collect this many macro-steps then PPO update
    low_level_deterministic: bool = False    # if low policy supports forcing mode actions


# =========================
# Helper: skill one-hot
# =========================

def one_hot(idx: int, dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    v[idx] = 1.0
    return v


def discounted_k_return(rewards: List[float], gamma: float) -> float:
    """Sum_{i=0}^{k-1} gamma^i r_{t+i}"""
    ret = 0.0
    g = 1.0
    for r in rewards:
        ret += g * float(r)
        g *= gamma
    return ret


# =========================
# Frozen encoder adaptor
# =========================

class FrozenLowLevelEncoder(nn.Module):
    """Reuse low-level policy encoder if available; else identity."""
    def __init__(self, low_level_policy: nn.Module, obs_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        # low_level_policy is typically PolicyEx from drq_url/networks.py
        mod = getattr(low_level_policy, "module", None)
        enc = getattr(mod, "encoder", None)
        self.encoder = enc if isinstance(enc, nn.Module) else None

        if self.encoder is not None:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.to(device)

            with torch.no_grad():
                dummy = torch.zeros(1, obs_dim, device=device)
                rep = self.encoder(dummy)
            self.rep_dim = int(rep.shape[-1])
        else:
            self.rep_dim = int(obs_dim)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return obs
        return self.encoder(obs)


# =========================
# Meta policy (discrete PPO)
# =========================

def mlp(in_dim: int, out_dim: int, hidden=(256, 256)) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class DiscreteMetaPolicy(nn.Module):
    """Categorical policy over discrete skills + value head."""
    def __init__(self, rep_dim: int, dim_skill: int):
        super().__init__()
        self.pi = mlp(rep_dim, dim_skill)
        self.v = mlp(rep_dim, 1)

    def forward(self, rep: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        logits = self.pi(rep)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.v(rep).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, rep: torch.Tensor) -> Tuple[int, float, float]:
        dist, v = self.forward(rep)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item()), float(v.item())

    @torch.no_grad()
    def value(self, rep: torch.Tensor) -> float:
        _, v = self.forward(rep)
        return float(v.item())


# =========================
# PPO buffer with variable macro-discount
# =========================

class MacroPPORollout:
    """
    Stores macro-transitions:
      (obs, act_idx, logp, value, macro_reward, macro_disc, done, next_obs)

    macro_reward = sum_{i=0}^{k_eff-1} gamma^i r_{t+i}
    macro_disc   = gamma^{k_eff}
    """
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.next_obs: List[np.ndarray] = []
        self.act: List[int] = []
        self.logp: List[float] = []
        self.val: List[float] = []
        self.rew: List[float] = []
        self.disc: List[float] = []
        self.done: List[bool] = []

    def add(self, obs, act, logp, val, rew, disc, done, next_obs):
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.act.append(int(act))
        self.logp.append(float(logp))
        self.val.append(float(val))
        self.rew.append(float(rew))
        self.disc.append(float(disc))
        self.done.append(bool(done))

    def __len__(self) -> int:
        return len(self.rew)

    def compute_gae(self, last_value: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Variable-discount GAE:
          delta_t = r_t + disc_t*(1-done_t)*V_{t+1} - V_t
          A_t = delta_t + disc_t*(1-done_t)*lam*A_{t+1}
        Returns (adv, ret) where ret = adv + V
        """
        n = len(self)
        adv = np.zeros(n, dtype=np.float32)
        next_adv = 0.0
        next_v = float(last_value)

        for t in reversed(range(n)):
            nonterminal = 0.0 if self.done[t] else 1.0
            delta = self.rew[t] + self.disc[t] * nonterminal * next_v - self.val[t]
            next_adv = delta + self.disc[t] * nonterminal * lam * next_adv
            adv[t] = next_adv
            next_v = self.val[t]

        ret = adv + np.asarray(self.val, dtype=np.float32)
        return adv, ret


# =========================
# Trainer
# =========================

class DiscreteMetaControllerPPOTrainer:
    """
    High-level meta-controller:
      z_t ~ pi_h(z|s_t) every k steps
    Low-level (frozen) METRA policy executes:
      a ~ pi_low(a|s,z)
    Reward for meta step is k-step discounted environment return.
    """

    def __init__(
            self,
            env,
            low_level_policy: nn.Module,
            dim_skill: int,
            device: torch.device,
            trainer_cfg: TrainerConfig = TrainerConfig(),
            ppo_cfg: MetaPPOConfig = MetaPPOConfig(),
    ):
        self.env = env
        self.low = low_level_policy
        self.dim_skill = int(dim_skill)
        self.device = device

        self.cfg = trainer_cfg
        self.ppo_cfg = ppo_cfg

        # Determine obs_dim
        ts = self.env.reset()
        obs0 = np.asarray(ts["image"]).reshape(-1).astype(np.float32)
        self.obs_dim = int(obs0.shape[0])

        # Freeze low-level
        self.low.eval()
        for p in self.low.parameters():
            p.requires_grad_(False)

        # Optional low-level deterministic switch (supported by drq_url metra policy)
        if hasattr(self.low, "_force_use_mode_actions"):
            self.low._force_use_mode_actions = bool(self.cfg.low_level_deterministic)

        # Encoder reuse (frozen)
        self.enc = FrozenLowLevelEncoder(self.low, obs_dim=self.obs_dim, device=self.device)

        self.render_size = getattr(getattr(self, "cfg", None), "render_size", None)
        self.framestack = getattr(getattr(self, "cfg", None), "framestack", None)
        self.obs_channels = 3

        # Meta policy
        self.meta = DiscreteMetaPolicy(rep_dim=self.enc.rep_dim, dim_skill=self.dim_skill).to(self.device)
        self.opt = torch.optim.Adam(self.meta.parameters(), lr=self.ppo_cfg.lr)

        self.total_env_steps = 0
        self.total_macro_steps = 0

    @torch.no_grad()
    def _encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.as_tensor(obs[None, :], device=self.device).float()
        rep = self.enc(obs_t)
        return rep  # (1, rep_dim)

    def _rollout_until(self, target_macro_steps: int) -> Tuple[MacroPPORollout, Dict[str, float]]:
        buf = MacroPPORollout()
        ep_returns: List[float] = []
        ep_steps: List[int] = []

        while len(buf) < target_macro_steps:
            ts = self.env.reset()
            obs = np.asarray(ts["image"]).reshape(-1).astype(np.float32)
            done = False
            steps = 0
            ep_ret = 0.0

            while not done and (self.cfg.max_episode_steps is None or steps < self.cfg.max_episode_steps):
                # meta chooses skill
                rep = self._encode_obs(obs)
                a_idx, logp, v = self.meta.act(rep)

                skill = one_hot(a_idx, self.dim_skill)  # discrete z
                # execute low-level for up to k steps
                rewards_k: List[float] = []
                k_eff = 0
                obs_start = obs.copy()

                for _ in range(self.cfg.k_steps):
                    ll_in = utils.get_np_concat_obs(obs, skill)  # concat (obs, z)

                    action, _info = self.low.get_action(ll_in)

                    ts = self.env.step({"action": action})
                    r = float(ts["reward"])
                    done = bool(ts["is_terminal"])
                    next_obs = np.asarray(ts["image"]).reshape(-1).astype(np.float32)

                    rewards_k.append(r)
                    ep_ret += r
                    steps += 1
                    self.total_env_steps += 1
                    k_eff += 1

                    obs = next_obs
                    if done:
                        break
                    if self.cfg.max_episode_steps is not None and steps >= self.cfg.max_episode_steps:
                        done = True
                        break

                macro_r = discounted_k_return(rewards_k, self.cfg.gamma)
                macro_disc = float(self.cfg.gamma ** k_eff)
                obs_next = obs.copy()

                buf.add(
                    obs=obs_start,
                    act=a_idx,
                    logp=logp,
                    val=v,
                    rew=macro_r,
                    disc=macro_disc,
                    done=done,
                    next_obs=obs_next,
                )
                self.total_macro_steps += 1

                if len(buf) >= target_macro_steps:
                    break

            ep_returns.append(ep_ret)
            ep_steps.append(steps)

        stats = {
            "episode_return_mean": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "episode_steps_mean": float(np.mean(ep_steps)) if ep_steps else 0.0,
        }
        return buf, stats

    def _ppo_update(self, buf: MacroPPORollout) -> Dict[str, float]:
        cfg = self.ppo_cfg
        n = len(buf)

        # Bootstrap last value from final next_obs if last transition not done
        last_done = buf.done[-1]
        if last_done:
            last_v = 0.0
        else:
            with torch.no_grad():
                rep_last = self._encode_obs(buf.next_obs[-1])
                last_v = self.meta.value(rep_last)

        adv, ret = buf.compute_gae(last_value=last_v, lam=cfg.gae_lambda)
        adv_t = torch.as_tensor(adv, device=self.device).float()
        ret_t = torch.as_tensor(ret, device=self.device).float()

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Prepare tensors (encode all obs in one batch)
        obs_np = np.asarray(buf.obs, dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, device=self.device).float()
        with torch.no_grad():
            rep_t = self.enc(obs_t)

        act_t = torch.as_tensor(np.asarray(buf.act, dtype=np.int64), device=self.device)
        logp_old_t = torch.as_tensor(np.asarray(buf.logp, dtype=np.float32), device=self.device)

        idx = np.arange(n)
        pi_losses, v_losses, entropies, kls = [], [], [], []

        for _ in range(cfg.train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]

                dist, v = self.meta(rep_t[mb])
                logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(logp - logp_old_t[mb])

                clip = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                loss_pi = -(torch.min(ratio * adv_t[mb], clip * adv_t[mb])).mean()

                loss_v = F.mse_loss(v, ret_t[mb])
                ent = dist.entropy().mean()

                loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (logp_old_t[mb] - logp).mean()

                pi_losses.append(loss_pi.item())
                v_losses.append(loss_v.item())
                entropies.append(ent.item())
                kls.append(approx_kl.item())

        return {
            "loss_pi": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "loss_v": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
        }

    def train(self, total_env_steps: int, log_every_updates: int = 1) -> None:
        update_idx = 0
        while self.total_env_steps < total_env_steps:
            buf, rollout_stats = self._rollout_until(self.cfg.rollout_macro_steps)
            update_stats = self._ppo_update(buf)
            update_idx += 1

            if update_idx % log_every_updates == 0:
                print(
                    f"[MetaPPO] update={update_idx} env_steps={self.total_env_steps} macro_steps={self.total_macro_steps} "
                    f"ep_ret_mean={rollout_stats['episode_return_mean']:.10f} "
                    f"loss_pi={update_stats['loss_pi']:.10f} loss_v={update_stats['loss_v']:.10f} "
                    f"ent={update_stats['entropy']:.10f} kl={update_stats['approx_kl']:.10f}"
                )

    @torch.no_grad()
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Deterministic meta policy: argmax over skills."""
        returns: List[float] = []
        steps_list: List[int] = []

        # force deterministic LL if supported
        if hasattr(self.low, "_force_use_mode_actions"):
            self.low._force_use_mode_actions = bool(self.cfg.low_level_deterministic)

        for _ in range(int(num_episodes)):
            ts = self.env.reset()
            obs = np.asarray(ts["image"]).reshape(-1).astype(np.float32)
            done = False
            ep_ret = 0.0
            steps = 0

            while not done and (self.cfg.max_episode_steps is None or steps < self.cfg.max_episode_steps):
                rep = self._encode_obs(obs)
                dist, _v = self.meta(rep)
                a_idx = int(torch.argmax(dist.logits, dim=-1).item())
                skill = one_hot(a_idx, self.dim_skill)

                for _ in range(self.cfg.k_steps):
                    ll_in = utils.get_np_concat_obs(obs, skill)
                    action, _info = self.low.get_action(ll_in)

                    ts = self.env.step({"action": action})
                    r = float(ts["reward"])
                    done = bool(ts["is_terminal"])
                    obs = np.asarray(ts["image"]).reshape(-1).astype(np.float32)

                    ep_ret += r
                    steps += 1
                    if done:
                        break
                    if self.cfg.max_episode_steps is not None and steps >= self.cfg.max_episode_steps:
                        done = True
                        break

            returns.append(ep_ret)
            steps_list.append(steps)

        return {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "steps_mean": float(np.mean(steps_list)) if steps_list else 0.0,
        }

    def _to_uint8_hwc(self, img):
        """Convert image to uint8 HWC format, allowing channels>=3."""
        import numpy as np

        if img is None:
            return None
        img = np.asarray(img)

        # CHW -> HWC if likely
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1, 2, 0))

        # grayscale -> RGB
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)

        if img.ndim != 3:
            return None
        if img.shape[0] < 2 or img.shape[1] < 2:
            return None

        # drop alpha if exactly 4 channels
        if img.shape[-1] == 4:
            img = img[..., :3]

        # dtype conversion
        if img.dtype != np.uint8:
            mx = float(img.max()) if img.size else 0.0
            if mx <= 1.0 + 1e-6:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        return img


    def _get_frame(self, obs_flat):
        """
        Recover a single RGB frame (H,W,3) from a flattened FrameStack observation.

        Works for drq_url env stack behavior:
          - Base env obs_space["image"].shape == (H, W, 3)
          - FrameStack obs_space["image"].shape == (H, W, 3*K)
          - rollout flattens timestep["image"] -> 1D
        """
        import numpy as np

        if obs_flat is None:
            return None

        x = np.asarray(obs_flat)

        # Case 1: already image-like
        if x.ndim in (2, 3):
            img = self._to_uint8_hwc(x)
            if img is None:
                return None
            # If stacked along channel dim (e.g., HWC with C=3*K), take last 3 channels
            if img.shape[-1] > 3 and (img.shape[-1] % 3 == 0):
                img = img[..., -3:]
            # ensure RGB
            if img.shape[-1] != 3:
                return None
            return img

        # Case 2: flattened -> reshape using env.obs_space["image"].shape
        if x.ndim != 1:
            return None

        # Use the *current env* obs_space (already includes framestack expansion)
        try:
            shp = tuple(self.env.obs_space["image"].shape)  # expected (H,W,3*K) even if flatten_obs=True
        except Exception:
            return None

        prod = int(np.prod(shp))
        if x.size < prod:
            return None

        # If extra stuff exists (rare for this codepath), take the first prod elements
        pix = x[:prod]

        # Most common: reshape to (H,W,3*K)
        try:
            stacked = pix.reshape(shp)
        except Exception:
            # Some implementations temporarily insert a leading 1 dim before flatten; try (1,H,W,3*K)
            try:
                stacked = pix.reshape((1,) + shp)[0]
            except Exception:
                return None

        img = self._to_uint8_hwc(stacked)
        if img is None:
            return None

        # Key fix: stacked channels are 3*K, so slice last 3 channels as "latest frame"
        if img.shape[-1] > 3 and (img.shape[-1] % 3 == 0):
            img = img[..., -3:]

        if img.shape[-1] != 3:
            return None

        return img


    @torch.no_grad()
    def rollout_eval_episode(self, record_video: bool = False) -> Dict[str, Any]:
        """
        Rollout one episode with deterministic meta-policy (argmax),
        returning:
          - episode_return: sum r_t
          - discounted_return: sum gamma^t r_t
          - frames: list of RGB frames (optional)
          - steps: number of env steps
        """
        # force deterministic LL if supported
        if hasattr(self.low, "_force_use_mode_actions"):
            self.low._force_use_mode_actions = bool(self.cfg.low_level_deterministic)

        ts = self.env.reset()
        obs = np.asarray(ts["image"]).reshape(-1).astype(np.float32)

        done = False
        ep_ret = 0.0
        ep_disc_ret = 0.0
        g = 1.0
        steps = 0
        frames = []

        # record initial frame if desired
        if record_video:
            f0 = self._get_frame(np.asarray(ts["image"]))
            if f0 is not None:
                frames.append(f0)

        while not done and (self.cfg.max_episode_steps is None or steps < self.cfg.max_episode_steps):
            # deterministic meta skill: argmax logits
            rep = self._encode_obs(obs)
            dist, _v = self.meta(rep)
            a_idx = int(torch.argmax(dist.logits, dim=-1).item())
            skill = one_hot(a_idx, self.dim_skill)

            # execute low-level for up to k steps
            for _ in range(self.cfg.k_steps):
                ll_in = utils.get_np_concat_obs(obs, skill)
                action, _info = self.low.get_action(ll_in)

                ts = self.env.step({"action": action})
                r = float(ts["reward"])
                done = bool(ts["is_terminal"])
                obs = np.asarray(ts["image"]).reshape(-1).astype(np.float32)

                ep_ret += r
                ep_disc_ret += g * r
                g *= self.cfg.gamma
                steps += 1

                if record_video:
                    fr = self._get_frame(np.asarray(ts["image"]))
                    if fr is not None:
                        frames.append(fr)

                if done:
                    break
                if self.cfg.max_episode_steps is not None and steps >= self.cfg.max_episode_steps:
                    done = True
                    break

        return {
            "episode_return": ep_ret,
            "discounted_return": ep_disc_ret,
            "frames": frames,
            "steps": steps,
        }

    @torch.no_grad()
    def evaluate_with_video(self, num_episodes: int = 16, video_dir: str = None, video_tag: str = "eval") -> Dict[str, float]:
        """
        Evaluate by rolling out num_episodes trajectories, compute:
          - return_mean / return_std
          - discounted_return_mean / discounted_return_std
          - avg_return_16 / avg_discounted_return_16 (same as means when num_episodes=16)
        Also randomly pick one trajectory and save to mp4 if video_dir is provided.
        """
        returns = []
        disc_returns = []
        steps_list = []
        traj_frames = []  # list of list-of-frames (only if recording)

        # Always collect frames for all episodes? No—only for a random one.
        # To avoid second rollout, we record frames for all and then pick one.
        # If you want less memory, do a second rollout just for video.
        record_all = True

        for _ in range(int(num_episodes)):
            out = self.rollout_eval_episode(record_video=record_all)
            returns.append(out["episode_return"])
            disc_returns.append(out["discounted_return"])
            steps_list.append(out["steps"])
            traj_frames.append(out["frames"])
        # trajectories = self._get_trajectories(num_episodes,
        #                                       deterministic_policy=True,
        #                                       )
        returns = np.asarray(returns, dtype=np.float32)
        disc_returns = np.asarray(disc_returns, dtype=np.float32)
        steps_list = np.asarray(steps_list, dtype=np.float32)

        metrics = {
            "return_mean": float(returns.mean()) if len(returns) else 0.0,
            "return_std": float(returns.std()) if len(returns) else 0.0,
            "discounted_return_mean": float(disc_returns.mean()) if len(disc_returns) else 0.0,
            "discounted_return_std": float(disc_returns.std()) if len(disc_returns) else 0.0,
            "avg_return_16": float(returns.mean()) if len(returns) else 0.0,
            "avg_discounted_return_16": float(disc_returns.mean()) if len(disc_returns) else 0.0,
            "steps_mean": float(steps_list.mean()) if len(steps_list) else 0.0,
        }

        # Save random trajectory video
        if video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)
            vid_idx = random.randint(0, int(num_episodes) - 1)
            frames = traj_frames[vid_idx]
            if frames is not None and len(frames) > 0:
                video_path = os.path.join(video_dir, f"{video_tag}_rand{vid_idx}.mp4")
                # fps: roughly env steps per second; you can adjust
                imageio.mimsave(video_path, frames, fps=30)
                metrics["video_path"] = video_path

        return metrics



