import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.mass.nn_mass import StreamingNNMass


def test_sparse_region_has_higher_reward_and_adds_reduce_it():
    random.seed(0)
    torch.manual_seed(0)
    dense = torch.zeros(100, 2)
    sparse = torch.full((5, 2), 10.0)
    data = torch.cat([dense, sparse], dim=0)

    mass = StreamingNNMass(
        z_dim=2,
        c=8,
        psi=32,
        short_size=300,
        long_size=300,
        w_short=0.7,
        w_long=0.3,
        device="cpu",
    )
    mass.build_initial_partitions(data)

    dense_reward = mass.reward_batch(torch.zeros(16, 2)).mean()
    sparse_query = torch.full((16, 2), 10.0)
    sparse_reward = mass.reward_batch(sparse_query).mean()
    assert sparse_reward > dense_reward

    before = mass.reward_batch(sparse_query).mean()
    mass.add_z(torch.full((120, 2), 10.0))
    after = mass.reward_batch(sparse_query).mean()
    assert after < before


def test_reward_batch_shape_repartition_and_weights():
    torch.manual_seed(1)
    data = torch.randn(50, 2)
    query = torch.randn(7, 2)
    mass = StreamingNNMass(
        z_dim=2,
        c=4,
        psi=8,
        short_size=100,
        long_size=100,
        w_short=0.7,
        w_long=0.3,
        device="cpu",
    )
    mass.build_initial_partitions(data)
    reward = mass.reward_batch(query)
    assert reward.shape == (7,)

    components = mass.reward_components(query)
    expected = 0.7 * components.short + 0.3 * components.long
    assert torch.allclose(components.total, expected)

    stats = mass.repartition()
    assert stats["short_size"] == 50.0
    assert stats["long_size"] == 50.0
    mass.rolling_refresh(1)
    assert mass.refresh_count == 1
