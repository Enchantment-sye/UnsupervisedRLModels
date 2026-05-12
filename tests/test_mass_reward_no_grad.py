import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.mass.coverage_encoder import CoverageEncoder
from core.mass.nn_mass import StreamingNNMass
from core.mass.reward_adapter import MassRewardAdapter


def test_mass_reward_adapter_does_not_create_coverage_encoder_grads():
    torch.manual_seed(0)
    encoder = CoverageEncoder(pixel_shape=(64, 64, 3), action_dim=2, latent_dim=8)
    mass = StreamingNNMass(z_dim=8, c=4, psi=8, short_size=100, long_size=100, device="cpu")
    mass.build_initial_partitions(torch.randn(64, 8))
    adapter = MassRewardAdapter(
        coverage_encoder=encoder,
        mass_model=mass,
        lambda_action=1e-3,
        lambda_delta_action=1e-3,
        lambda_done=5.0,
        device="cpu",
    )

    batch = {
        "next_obs": torch.randint(0, 256, (5, 64 * 64 * 3), dtype=torch.uint8).float(),
        "actions": torch.randn(5, 2),
        "prev_actions": torch.randn(5, 2),
        "dones": torch.zeros(5),
    }
    with torch.enable_grad():
        out = adapter.compute_batch_reward(batch, update_rms=True)

    assert out["r_cov"].requires_grad is False
    assert out["rewards"].requires_grad is False
    assert all(param.grad is None for param in encoder.parameters())


def test_mass_reward_adapter_chunked_encoding_matches_full_batch():
    torch.manual_seed(1)
    encoder = CoverageEncoder(pixel_shape=(64, 64, 3), action_dim=2, latent_dim=8)
    mass = StreamingNNMass(z_dim=8, c=4, psi=8, short_size=100, long_size=100, device="cpu")
    mass.build_initial_partitions(torch.randn(64, 8))
    batch = {
        "next_obs": torch.randint(0, 256, (7, 64 * 64 * 3), dtype=torch.uint8).float(),
        "actions": torch.randn(7, 2),
        "prev_actions": torch.randn(7, 2),
        "dones": torch.zeros(7),
    }

    full = MassRewardAdapter(
        coverage_encoder=encoder,
        mass_model=mass,
        encode_batch_size=99,
        device="cpu",
    ).compute_batch_reward(batch, update_rms=False)
    chunked = MassRewardAdapter(
        coverage_encoder=encoder,
        mass_model=mass,
        encode_batch_size=2,
        device="cpu",
    ).compute_batch_reward(batch, update_rms=False)

    assert full["rewards"].shape == (7,)
    assert torch.allclose(full["r_cov"], chunked["r_cov"], atol=1e-6)
    assert torch.allclose(full["rewards"], chunked["rewards"], atol=1e-6)
