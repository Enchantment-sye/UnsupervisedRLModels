import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core import sac_utils
from core.metra_agent import MetraAgent
from core.metra_variants import BaseVariant, DiaynVariant
from core.sac_trainer import SacTrainer
from utils import utils


REQUIRED_KEYS = {
    "PureRewardMean",
    "PureRewardStd",
    "PureRewardMin",
    "PureRewardMax",
    "ScaledRewardMean",
    "ScaledRewardStd",
    "DeltaPhiNormMean",
    "DeltaPhiNormStd",
    "DeltaPhiNormMax",
    "Q1Mean",
    "Q2Mean",
    "QTargetsMean",
    "QTargetsStd",
    "QTdErrAbsMean",
    "LossSacp",
    "SacpNewActionLogProbMean",
    "Alpha",
    "LogAlpha",
    "AlphaLr",
    "LossAlpha",
    "DualLam",
    "LossDualLam",
    "DualCstPenalty",
    "TemporalViolationMean",
    "TemporalViolationFrac",
    "TotalGradNormAll",
    "TotalGradNormTrajEncoder",
    "TotalGradNormOptionPolicy",
    "TotalGradNormQf",
    "TotalGradNormDualLam",
    "TotalGradNormLogAlpha",
}


class _ParamModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.param = torch.nn.Parameter(torch.as_tensor([value], dtype=torch.float32))


class _Policy(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.linear = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, obs):
        mean = self.linear(obs)
        std = torch.ones_like(mean)
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)


class _Qf(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.linear = torch.nn.Linear(obs_dim + action_dim, 1)

    def forward(self, obs, actions):
        return self.linear(torch.cat([obs, actions], dim=-1))


class _TrajEncoder(torch.nn.Module):
    def __init__(self, obs_dim, phi_dim):
        super().__init__()
        self.linear = torch.nn.Linear(obs_dim, phi_dim)

    def forward(self, obs):
        mean = self.linear(obs)
        std = torch.ones_like(mean)
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)


def _cfg(*, dual_reg, inner=True, discrete=False, algo="metra"):
    cascade = SimpleNamespace(use_cascade=False)
    cfg = SimpleNamespace(
        algo=algo,
        use_target_traj_encoder=False,
        use_kme=False,
        use_novelty_reward=False,
        kernel_map=False,
        dual_dist="one",
        dual_reg=dual_reg,
        dual_slack=10.0,
        inner=inner,
        discrete=discrete,
        dim_skill=2,
        use_hierarchical_skill=False,
        num_skill_levels=1,
        cascade=cascade,
        use_hierarchical_phi=False,
        beta_mode="uniform",
        beta_rho=0.5,
        log_beta_values=True,
    )
    cfg.algo = SimpleNamespace(
        algo=algo,
        use_target_traj_encoder=False,
        use_kme=False,
        kernel_map=False,
        dual_dist=cfg.dual_dist,
        dual_reg=dual_reg,
        dual_slack=cfg.dual_slack,
        inner=inner,
        discrete=discrete,
        dim_skill=2,
        use_hierarchical_skill=False,
        num_skill_levels=1,
    )
    return cfg


def _fake_active_agent(*, dual_reg, inner=True, discrete=False, variant_cls=BaseVariant, batch_size=2):
    obs_dim = 3
    phi_dim = 2
    cfg = _cfg(dual_reg=dual_reg, inner=inner, discrete=discrete, algo="diayn" if variant_cls is DiaynVariant else "metra")
    traj_encoder = _TrajEncoder(obs_dim, phi_dim)
    dual_lam = _ParamModule(torch.log(torch.tensor(30.0)).item())
    agent = SimpleNamespace(
        cfg=cfg,
        device=torch.device("cpu"),
        traj_encoder=traj_encoder,
        target_traj_encoder=traj_encoder,
        traj_latent_normalizer=None,
        target_traj_latent_normalizer=None,
        dual_lam=dual_lam,
        variant=variant_cls(cfg),
    )
    agent.phi_from_encoder_output = MetraAgent.phi_from_encoder_output.__get__(agent)
    agent.normalize_phi_tensor = MetraAgent.normalize_phi_tensor.__get__(agent)
    agent._get_phi_normalizer = MetraAgent._get_phi_normalizer.__get__(agent)
    agent._optimizer = utils.OptimizerGroupWrapper(
        {
            "traj_encoder": torch.optim.Adam(traj_encoder.parameters(), lr=1e-4),
            "dual_lam": torch.optim.Adam(dual_lam.parameters(), lr=1e-4),
        }
    )
    agent._ensure_train_diagnostic_defaults = MetraAgent._ensure_train_diagnostic_defaults.__get__(agent)
    agent._record_grad_norms = MetraAgent._record_grad_norms.__get__(agent)
    agent._gradient_descent = MetraAgent._gradient_descent.__get__(agent)
    agent._optimize_te = MetraAgent._optimize_te.__get__(agent)

    if discrete:
        skills = torch.eye(phi_dim).repeat((batch_size + phi_dim - 1) // phi_dim, 1)[:batch_size]
    else:
        base_skills = torch.tensor([[0.5, -0.2], [0.1, 0.4]], dtype=torch.float32)
        skills = base_skills.repeat((batch_size + 1) // 2, 1)[:batch_size]
    batch = {
        "obs": torch.zeros(batch_size, 3),
        "next_obs": torch.ones(batch_size, 3) * 0.25,
        "skills": skills,
    }
    return agent, batch


def _add_sac_metrics(metrics, batch_size=4):
    obs_dim = 3
    action_dim = 2
    obs = torch.randn(batch_size, obs_dim)
    next_obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)

    qf1 = _Qf(obs_dim, action_dim)
    qf2 = _Qf(obs_dim, action_dim)
    target_qf1 = _Qf(obs_dim, action_dim)
    target_qf2 = _Qf(obs_dim, action_dim)
    policy = _Policy(obs_dim, action_dim)
    log_alpha = _ParamModule(torch.log(torch.tensor(0.01)).item())
    optimizer = utils.OptimizerGroupWrapper(
        {
            "actor": torch.optim.Adam(policy.parameters(), lr=1e-4),
            "qf": torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=1e-4),
            "log_alpha": torch.optim.Adam(log_alpha.parameters(), lr=0.0),
        }
    )
    algo = SimpleNamespace(
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        log_alpha=log_alpha,
        discount=0.99,
        tau=0.005,
        _target_entropy=-float(action_dim),
        _env_spec=None,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )

    sac_utils.update_loss_qf(
        algo,
        metrics,
        {},
        obs,
        actions,
        next_obs,
        dones,
        rewards,
        policy,
    )
    v = {}
    sac_utils.update_loss_sacp(algo, metrics, v, obs, policy)
    sac_utils.update_loss_alpha(algo, metrics, v)

    for loss, keys in (
        (metrics["LossQf1"] + metrics["LossQf2"], ["qf"]),
        (metrics["LossSacp"], ["actor"]),
        (metrics["LossAlpha"], ["log_alpha"]),
    ):
        optimizer.zero_grad(keys=keys)
        loss.backward(retain_graph=True)
        SacTrainer._record_grad_norms(algo, metrics, keys)


def _assert_required_keys(metrics):
    missing = REQUIRED_KEYS.difference(metrics)
    assert not missing
    for key in REQUIRED_KEYS:
        value = metrics[key]
        assert torch.is_tensor(value), key
        assert value.numel() == 1, key


def test_train_metric_keys_exist_with_dual_reg_true():
    metrics = {}
    agent, batch = _fake_active_agent(dual_reg=True)

    agent._optimize_te(metrics, batch)
    agent.variant.compute_intrinsic_reward(agent, batch, metrics)
    _add_sac_metrics(metrics)

    _assert_required_keys(metrics)
    assert metrics["TotalGradNormDualLam"].item() > 0.0
    assert metrics["AlphaLr"].item() == 0.0
    assert metrics["Alpha"].item() == torch.tensor(0.01).item()


def test_train_metric_keys_exist_with_dual_reg_false():
    metrics = {}
    agent, batch = _fake_active_agent(dual_reg=False)

    agent._optimize_te(metrics, batch)
    agent.variant.compute_intrinsic_reward(agent, batch, metrics)
    _add_sac_metrics(metrics)

    _assert_required_keys(metrics)


def test_train_metric_keys_exist_for_inner_false_discrete():
    metrics = {}
    agent, batch = _fake_active_agent(
        dual_reg=False,
        inner=False,
        discrete=True,
        variant_cls=DiaynVariant,
    )

    agent._optimize_te(metrics, batch)
    agent.variant.compute_intrinsic_reward(agent, batch, metrics)

    for key in ("PureRewardMean", "DeltaPhiNormMean", "DualLam", "TemporalViolationFrac", "TotalGradNormTrajEncoder"):
        assert key in metrics


def test_new_std_diagnostics_are_finite_with_batch_size_one():
    metrics = {}
    agent, batch = _fake_active_agent(dual_reg=False, batch_size=1)

    agent._optimize_te(metrics, batch)
    agent.variant.compute_intrinsic_reward(agent, batch, metrics)
    _add_sac_metrics(metrics, batch_size=1)

    for key in ("PureRewardStd", "DeltaPhiNormStd", "QTargetsStd", "ScaledRewardStd"):
        assert key in metrics
        assert not torch.isnan(metrics[key]), key
