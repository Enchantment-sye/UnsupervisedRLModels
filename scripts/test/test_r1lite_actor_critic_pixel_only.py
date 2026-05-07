import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from core import sac_utils
from utils.agent_utils import process_samples


class _TinyAlgo:
    def __init__(self):
        self.device = torch.device("cpu")
        self.log_alpha = type("LogAlpha", (), {"param": torch.zeros(1)})()
        self._env_spec = None
        self.safe_action_distill_weight = 0.5
        self.qf1 = None
        self.qf2 = None


class _Policy(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        batch = obs.shape[0]
        return torch.distributions.Independent(
            torch.distributions.Normal(self.mean.expand(batch, -1), torch.ones(batch, self.mean.numel())),
            1,
        )


class _Q(torch.nn.Module):
    def forward(self, obs, action):
        return -torch.sum(action ** 2, dim=-1, keepdim=True)


def test_process_samples_keeps_safety_state_out_of_obs_and_adds_optional_metrics():
    obs = np.zeros((2, 2 * 2 * 9), dtype=np.uint8)
    actions = np.ones((2, 14), dtype=np.float32)
    path = {
        "observations": obs,
        "next_observations": obs.copy(),
        "actions": actions,
        "rewards": np.zeros(2, dtype=np.float32),
        "dones": np.asarray([False, True]),
        "agent_infos": {
            "skill": np.zeros((2, 8), dtype=np.float32),
            "raw_action": actions + 1.0,
            "safe_action": actions,
            "safety_correction_norm": np.ones(2, dtype=np.float32),
        },
        "env_infos": {
            "safety_min_margin": np.ones(2, dtype=np.float32),
            "safety_redline_count": np.zeros(2, dtype=np.float32),
        },
    }

    data = process_samples([path], discount=0.99)

    assert data["obs"][0].shape == obs.shape
    assert "safety_state" not in data
    assert "raw_actions" in data
    assert "safe_actions" in data
    assert "safety_correction_norm" in data


def test_sac_safe_action_distillation_uses_safe_actions_without_proprio_obs():
    algo = _TinyAlgo()
    metrics = {}
    v = {"safe_actions": torch.zeros(4, 3)}
    obs = torch.randn(4, 16)
    policy = _Policy(action_dim=3)
    qf = _Q()

    sac_utils.update_loss_sacp(algo, metrics, v, obs=obs, policy=policy, qf1=qf, qf2=qf)

    assert "LossSafeActionDistill" in metrics
    assert v["new_actions"].shape == (4, 3)
    assert obs.shape == (4, 16)
