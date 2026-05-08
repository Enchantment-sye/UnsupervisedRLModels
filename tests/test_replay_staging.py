import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("src"))

from core.metra_agent import MetraAgent
from core.metra_trainer import MetraTrainer
from utils import agent_utils
from utils import utils


def test_replay_tensor_stager_matches_numpy_batch_to_torch_on_cpu():
    value = np.arange(12, dtype=np.float32).reshape(3, 4)
    stager = agent_utils.ReplayTensorStager(pin_memory=True)

    staged, stage_sec, h2d_sec = stager.to_torch("obs", value, torch.device("cpu"))
    direct = agent_utils.numpy_batch_to_torch(value, torch.device("cpu"), dtype=torch.float32)

    assert torch.equal(staged, direct)
    assert staged.device.type == "cpu"
    assert staged.shape == direct.shape
    assert staged.dtype == direct.dtype
    assert stage_sec >= 0.0
    assert h2d_sec >= 0.0


class _ReplayBuffer:
    def __init__(self, samples):
        self.samples = samples

    def sample_transitions(self, batch_size):
        assert batch_size == 3
        return {key: value.copy() for key, value in self.samples.items()}


def test_sample_replay_buffer_staging_preserves_values_and_shapes():
    samples = {
        "obs": np.arange(6, dtype=np.float32).reshape(3, 2),
        "next_obs": np.arange(6, 12, dtype=np.float32).reshape(3, 2),
        "rewards": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        "skills": np.ones((3, 1), dtype=np.float32),
    }
    agent = MetraAgent.__new__(MetraAgent)
    agent.cfg = SimpleNamespace(
        train=SimpleNamespace(trans_minibatch_size=3),
        replay_staging_enabled=True,
        replay_staging_pin_memory=True,
    )
    agent.device = torch.device("cpu")
    agent.replay_buffer = _ReplayBuffer(samples)
    agent._replay_tensor_stager = None

    batch = MetraAgent._sample_replay_buffer(agent)

    assert torch.equal(batch["obs"], torch.as_tensor(samples["obs"]))
    assert torch.equal(batch["next_obs"], torch.as_tensor(samples["next_obs"]))
    assert torch.equal(batch["rewards"], torch.as_tensor([1.0, 2.0, 3.0]))
    assert batch["skills"].shape == (3, 1)
    assert float(agent._last_replay_sample_seconds) >= 0.0
    assert float(agent._last_replay_stage_seconds) >= 0.0
    assert float(agent._last_replay_h2d_seconds) >= 0.0
    assert float(agent._last_replay_transfer_seconds) >= float(agent._last_replay_sample_seconds)


def test_train_once_skips_flatten_data_when_using_replay(monkeypatch):
    path = {
        "observations": np.zeros((2, 2), dtype=np.float32),
        "next_observations": np.ones((2, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.ones(2, dtype=np.float32),
        "dones": np.asarray([False, True]),
        "agent_infos": {"skill": np.zeros((2, 2), dtype=np.float32)},
        "env_infos": {},
    }
    trainer = MetraTrainer.__new__(MetraTrainer)
    trainer.cfg = SimpleNamespace(
        sac_discount=0.99,
        sac_min_buffer_size=1,
        kernel_map=False,
        stage="pre_training",
        algo=SimpleNamespace(algo="metra", use_kme=False),
        env=SimpleNamespace(task="ant"),
        safety=None,
    )
    trainer.replay_buffer = SimpleNamespace(n_transitions_stored=2)
    trainer.coverage_tracker = SimpleNamespace(update_train_paths=lambda paths: None)
    trainer.env = SimpleNamespace(spec=object())
    trainer.step_itr = 0
    trainer._log_metrics = lambda *args, **kwargs: None
    trainer.agent = SimpleNamespace(
        device=torch.device("cpu"),
        _update_replay_buffer=lambda data: None,
        update=lambda epoch_data: {"UpdateCalled": 1.0} if epoch_data is None else (_ for _ in ()).throw(AssertionError("expected replay update with epoch_data=None")),
    )
    monkeypatch.setattr(agent_utils, "flatten_data", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("flatten_data should be skipped")))
    monkeypatch.setattr(
        utils,
        "log_performance_ex",
        lambda *args, **kwargs: {
            "undiscounted_returns": [2.0],
            "discounted_returns": [2.0],
            "scalars": {},
            "histograms": {},
        },
    )

    result = trainer.train_once([path])

    assert result == 2.0
