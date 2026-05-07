import collections
import copy
import os
import tempfile
import time
from math import sqrt
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from core.metra_agent import MetraAgent
from core.metra_config import get_parser, make_config_from_args
from core.metra_logging import eval_log_diagnostics, log_diagnostics, plot_log_diagnostics, setup_logger
from core.metra_trainer import MeasureAndAccTime, MetraTrainer
from models.encoders import WithEncoder
from utils import utils


def legacy_kwargs_to_config(legacy_kwargs: Dict[str, Any]):
    """Map the legacy DRQ_METRAAgent constructor kwargs onto MetraConfig."""

    parser = get_parser()
    args = parser.parse_args([])

    mapping = {
        "env_name": "task",
        "tau": "sac_tau",
        "scale_reward": "sac_scale_reward",
        "target_coef": "sac_target_coef",
        "min_buffer_size": "sac_min_buffer_size",
        "discount": "sac_discount",
        "use_encoder": "encoder",
        "actor_critic_backbone": "ac_backbone",
    }

    for key, value in legacy_kwargs.items():
        arg_name = mapping.get(key, key)
        if hasattr(args, arg_name):
            setattr(args, arg_name, value)

    if hasattr(args, "encoder"):
        args.encoder = int(bool(args.encoder))
    if hasattr(args, "sample_cpu"):
        args.sample_cpu = int(bool(args.sample_cpu))
    if hasattr(args, "discrete"):
        args.discrete = int(bool(args.discrete))
    if hasattr(args, "unit_length"):
        args.unit_length = int(bool(args.unit_length))
    if hasattr(args, "dual_reg"):
        args.dual_reg = int(bool(args.dual_reg))
    if hasattr(args, "spectral_normalization"):
        args.spectral_normalization = int(bool(args.spectral_normalization))

    return make_config_from_args(args)


class LegacyMetraAdapter:
    """Compatibility layer that recreates the legacy METRA API over the new core stack."""

    def __init__(self, **legacy_kwargs):
        self._legacy_kwargs = dict(legacy_kwargs)
        self._env = legacy_kwargs["env"]
        self._env_spec = legacy_kwargs["env_spec"]
        self.replay_buffer = legacy_kwargs["replay_buffer"]
        self.snapshot_dir = legacy_kwargs.get("snapshot_dir") or tempfile.mkdtemp(prefix="metra_compat_")

        self.cfg = legacy_kwargs_to_config(legacy_kwargs)
        self._agent = MetraAgent(self.cfg, self._env, self.replay_buffer)

        self.env_name = legacy_kwargs["env_name"]
        self.algo = legacy_kwargs["algo"]
        self.seed = legacy_kwargs.get("seed", 0)
        self.enable_logging = True
        self.step_itr = 0
        self.total_train_steps = 0
        self.total_env_steps = 0
        self.total_epoch = 0
        self._start_time = time.time()
        self._itr_start_time = self._start_time

        self.policy_delay = int(legacy_kwargs.get("policy_delay", 1))
        self.actor_start_steps = int(legacy_kwargs.get("actor_start_steps", 0))
        self.discount = float(legacy_kwargs.get("discount", self.cfg.train.sac_discount))
        self.time_limit = legacy_kwargs["time_limit"]
        self.device = legacy_kwargs.get("device", torch.device("cpu"))
        self.sample_cpu = legacy_kwargs.get("sample_cpu", True)
        self.stage = legacy_kwargs.get("stage", "pre_training")
        self.alpha = float(legacy_kwargs.get("alpha", self.cfg.algo.alpha))
        self.name = legacy_kwargs.get("name", "IOD")
        self.grad_clip_norm = legacy_kwargs.get("grad_clip_norm")

        self.dim_skill = int(legacy_kwargs["dim_skill"])
        self.discrete = bool(legacy_kwargs.get("discrete", False))
        self.unit_length = bool(legacy_kwargs.get("unit_length", False))
        self.inner = bool(legacy_kwargs.get("inner", self.cfg.algo.inner))
        self.batch_size = int(legacy_kwargs.get("batch_size", 32))
        self.tau = float(legacy_kwargs.get("tau", self.cfg.train.sac_tau))
        self.dual_reg = bool(legacy_kwargs.get("dual_reg", False))
        self.dual_slack = float(legacy_kwargs.get("dual_slack", self.cfg.algo.dual_slack))
        self.dual_dist = legacy_kwargs.get("dual_dist", self.cfg.algo.dual_dist)
        self.use_target_traj_encoder = bool(legacy_kwargs.get("use_target_traj_encoder", False))
        self.use_kme = bool(legacy_kwargs.get("use_kme", False))
        self.use_novelty_reward = bool(legacy_kwargs.get("use_novelty_reward", False))
        self.kernel_map_obs = bool(legacy_kwargs.get("kernel_map", False))
        self.idk_subsample_size = int(legacy_kwargs.get("idk_subsample_size", 256))
        self.idk_init = legacy_kwargs.get("idk_init", "gaussian")
        self.idk_from = legacy_kwargs.get("idk_from", "traj")
        self.idk_groups = int(legacy_kwargs.get("idk_groups", 1))
        self.min_buffer_size = int(legacy_kwargs.get("min_buffer_size", self.cfg.train.sac_min_buffer_size))
        self.num_alt_samples = int(legacy_kwargs.get("num_alt_samples", self.cfg.algo.num_alt_samples))
        self.split_group = int(legacy_kwargs.get("split_group", self.cfg.algo.split_group))

        self.n_epochs_per_eval = int(legacy_kwargs.get("n_epochs_per_eval", self.cfg.log.n_epochs_per_eval))
        self.n_epochs_per_log = int(legacy_kwargs.get("n_epochs_per_log", self.cfg.log.n_epochs_per_log))
        self.n_epochs_per_tb = int(legacy_kwargs.get("n_epochs_per_tb", self.cfg.log.n_epochs_per_tb))
        self.n_epochs_per_save = int(legacy_kwargs.get("n_epochs_per_save", self.cfg.log.n_epochs_per_save))
        self.n_epochs_per_pt_save = int(legacy_kwargs.get("n_epochs_per_pt_save", self.cfg.log.n_epochs_per_pt_save))
        self.n_epochs_per_pkl_update = legacy_kwargs.get("n_epochs_per_pkl_update", self.cfg.log.n_epochs_per_pkl_update)
        self.num_random_trajectories = int(legacy_kwargs.get("num_random_trajectories", self.cfg.log.num_random_trajectories))
        self.num_video_repeats = int(legacy_kwargs.get("num_video_repeats", self.cfg.log.num_video_repeats))
        self.eval_record_video = int(legacy_kwargs.get("eval_record_video", self.cfg.log.eval_record_video))
        self.video_skip_frames = int(legacy_kwargs.get("video_skip_frames", self.cfg.log.video_skip_frames))
        self.eval_plot_axis = legacy_kwargs.get("eval_plot_axis", self.cfg.log.eval_plot_axis)

        self._trans_minibatch_size = int(legacy_kwargs.get("trans_minibatch_size", self.cfg.train.trans_minibatch_size))
        self._trans_optimization_epochs = int(legacy_kwargs.get("trans_optimization_epochs", self.cfg.train.trans_optimization_epochs))

        self.writer = None
        self.logger = None

    def __getattr__(self, name):
        if name == "_agent":
            raise AttributeError(name)
        return getattr(self._agent, name)

    def _get_concat_obs(self, obs, skill):
        if skill is None:
            if int(getattr(self, "dim_skill", 0)) == 0:
                return obs
            raise ValueError("[METRA] skill is None but dim_skill > 0.")
        return utils.get_torch_concat_obs(obs, skill)

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(
                np.concatenate(value, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
        return epoch_data

    def process_samples(self, paths):
        data = collections.defaultdict(list)
        for path in paths:
            data["obs"].append(path["observations"])
            data["next_obs"].append(path["next_observations"])
            data["actions"].append(path["actions"])
            data["rewards"].append(path["rewards"])
            data["dones"].append(path["dones"])
            data["returns"].append(utils.discount_cumsum(path["rewards"], self.discount))
            if "pre_tanh_value" in path["agent_infos"]:
                data["pre_tanh_values"].append(path["agent_infos"]["pre_tanh_value"])
            if "log_prob" in path["agent_infos"]:
                data["log_probs"].append(path["agent_infos"]["log_prob"])
            if "skill" in path["agent_infos"]:
                data["skills"].append(path["agent_infos"]["skill"])
                data["next_skills"].append(
                    np.concatenate([path["agent_infos"]["skill"][1:], path["agent_infos"]["skill"][-1:]], axis=0)
                )
        return data

    def _update_replay_buffer(self, data):
        return self._agent._update_replay_buffer(data)

    def _sample_replay_buffer(self):
        return self._agent._sample_replay_buffer()

    def _gradient_descent(self, loss, optimizer_keys):
        return self._agent._gradient_descent(loss, optimizer_keys)

    def all_parameters(self):
        return self._agent.all_parameters()

    def _get_encoder_maps(self, v):
        cur_z, next_z = v.get("cur_z"), v.get("next_z")
        if cur_z is None:
            obs = v["obs"]
            next_obs = v["next_obs"]
            cur_z = self.traj_encoder(obs)
            next_z = self.traj_encoder(next_obs)
            v.update({"cur_z": cur_z, "next_z": next_z})
        return v.get("cur_z"), v.get("next_z"), v.get("skills")

    def _get_kernel_maps(self, v):
        kernel_cur_z = v.get("kernel_cur_z")
        kernel_next_z = v.get("kernel_next_z")
        kernel_skills = v.get("kernel_skills")

        if kernel_cur_z is None:
            traj_encoder = self.target_traj_encoder if self.use_target_traj_encoder else self.traj_encoder
            obs = v["obs"]
            next_obs = v["next_obs"]
            cur_z = traj_encoder(obs)
            next_z = traj_encoder(next_obs)
            kernel_cur_z = self.kernel(cur_z.mean) / sqrt(self.kernel.ensemble_size)
            kernel_next_z = self.kernel(next_z.mean) / sqrt(self.kernel.ensemble_size)
            kernel_skills = self.kernel(v["skills"]) / sqrt(self.kernel.ensemble_size)
            v.update(
                {
                    "kernel_cur_z": kernel_cur_z,
                    "kernel_next_z": kernel_next_z,
                    "kernel_skills": kernel_skills,
                    "cur_z": cur_z,
                    "next_z": next_z,
                }
            )

        return v.get("kernel_cur_z"), v.get("kernel_next_z"), v.get("kernel_skills")

    def _get_batch_emb_vectors(self, v):
        if self.use_kme and self.kernel_map_obs:
            return self._get_kernel_maps(v)
        return self._get_encoder_maps(v)

    def _update_distributional_novelty_rewards(self, metrics, v):
        cur_z, next_z, skills = self._get_kernel_maps(v)
        _ = cur_z, skills
        rewards = torch.matmul(next_z, self.kme_vector)
        rewards = torch.clamp(-torch.log(rewards), min=1, max=5)
        metrics.update({"NoveltyRewardMean": rewards.mean(), "NoveltyRewardStd": rewards.std()})
        v["novelty_rewards"] = rewards

    def _update_skill_rewards(self, metrics, v):
        cur_z, next_z, skills = self._get_batch_emb_vectors(v)

        if self.inner:
            rewards = torch.sum((next_z.mean - cur_z.mean) * skills, dim=1)
        else:
            goal = next_z.mean
            logits = torch.matmul(goal, skills.T)
            rewards = torch.diag(logits) - torch.logsumexp(logits, dim=1)

        v["skill_rewards"] = rewards
        metrics.update({"PureRewardMean": rewards.mean(), "PureRewardStd": rewards.std()})

    def _update_rewards(self, metrics, v):
        self._update_skill_rewards(metrics, v)
        if self.use_kme and self.use_novelty_reward:
            self._update_distributional_novelty_rewards(metrics, v)
        rewards = v["skill_rewards"] * v.get("novelty_rewards", 1)
        v["rewards"] = rewards
        metrics.update({"RewardMean": rewards.mean(), "RewardStd": rewards.std()})

    def _update_loss_dual_lam(self, metrics, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * v["cst_penalty"].detach().mean()
        metrics.update({"DualLam": dual_lam, "LossDualLam": loss_dual_lam})

    def _update_loss_te(self, metrics, v):
        self._update_rewards(metrics, v)
        rewards = v["rewards"]
        obs = v["obs"]
        next_obs = v["next_obs"]
        cur_z, next_z = v["cur_z"], v["next_z"]

        metrics.update(
            {
                "currentStateMean": torch.square(cur_z.mean).mean(),
                "currentStateStd": torch.norm(cur_z.mean).std(),
            }
        )

        if self.dual_dist == "s2_from_s" and self.dist_predictor is not None:
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            metrics["LossDp"] = loss_dp

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x, phi_y, _skills = self._get_batch_emb_vectors(v)

            if self.dual_dist == "l2":
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == "one":
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == "s2_from_s":
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1.0 / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)
                metrics.update(
                    {
                        "ScalingFactor": scaling_factor.mean(dim=0),
                        "NormalizedScalingFactor": normalized_scaling_factor.mean(dim=0),
                    }
                )
            elif self.use_kme and self.dual_dist == "skill_kme":
                cst_dist = 1e-6 * torch.einsum("ij,ij->i", (phi_x + phi_y) / 2, v["skill_kme"]).unsqueeze(0)
            elif self.dual_dist == "kernel_mmd":
                kernel_state, kernel_next_state, _kernel_skills = self._get_kernel_maps(v)
                cst_dist = torch.square(kernel_next_state - kernel_state).mean(dim=1)
            elif self.dual_dist == "kernel_sim_dist":
                kernel_state, kernel_next_state, _kernel_skills = self._get_kernel_maps(v)
                cst_dist = 1 - (kernel_next_state * kernel_state).sum(dim=1)
            elif self.dual_dist == "kernel_sim":
                kernel_state, kernel_next_state, _kernel_skills = self._get_kernel_maps(v)
                cst_dist = (kernel_next_state * kernel_state).sum(dim=1)
            else:
                raise NotImplementedError(f"Unsupported dual_dist={self.dual_dist!r}")

            metrics.update({"OriginalCstDist": cst_dist.mean(), "OriginalCstStd": cst_dist.std()})
            if self.dual_dist != "kernel_sim":
                cst_penalty = cst_dist - torch.square(phi_y.mean - phi_x.mean).mean(dim=1)
            else:
                cst_penalty = (phi_y.mean - phi_x.mean).sum(dim=1) - cst_dist
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            v["cst_penalty"] = cst_penalty
            metrics["DualCstPenalty"] = cst_penalty.mean()
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()
        metrics.update({"TeObjMean": te_obj.mean(), "LossTe": loss_te})

    def _optimize_te(self, metrics, internal_vars):
        self._update_loss_te(metrics, internal_vars)
        self._gradient_descent(metrics["LossTe"], optimizer_keys=["traj_encoder"])

        if self.dual_reg:
            self._update_loss_dual_lam(metrics, internal_vars)
            self._gradient_descent(metrics["LossDualLam"], optimizer_keys=["dual_lam"])
            if self.dual_dist == "s2_from_s" and "LossDp" in metrics:
                self._gradient_descent(metrics["LossDp"], optimizer_keys=["dist_predictor"])

    def _optimize_op(self, metrics, internal_vars, step):
        batch_size = internal_vars["obs"].shape[0]
        device = internal_vars["obs"].device

        if "skills" not in internal_vars or internal_vars["skills"] is None:
            internal_vars["skills"] = torch.zeros((batch_size, 0), device=device, dtype=torch.float32)
        if "next_skills" not in internal_vars or internal_vars["next_skills"] is None:
            internal_vars["next_skills"] = torch.zeros((batch_size, 0), device=device, dtype=torch.float32)
        if "obs_aug" in internal_vars and ("skills_aug" not in internal_vars or internal_vars["skills_aug"] is None):
            internal_vars["skills_aug"] = internal_vars["skills"]
        if "next_obs_aug" in internal_vars and (
            "next_skills_aug" not in internal_vars or internal_vars["next_skills_aug"] is None
        ):
            internal_vars["next_skills_aug"] = internal_vars["next_skills"]

        processed_obs = self.sac_trainer.skill_policy.process_observations(internal_vars["obs"])
        next_processed_obs = self.sac_trainer.skill_policy.process_observations(internal_vars["next_obs"])
        processed_cat_obs = self._get_concat_obs(processed_obs, internal_vars["skills"])
        next_processed_cat_obs = self._get_concat_obs(next_processed_obs, internal_vars["next_skills"])

        self.sac_trainer._optimize_once(metrics, internal_vars, processed_cat_obs, next_processed_cat_obs, step)

    @torch.no_grad()
    def update_target_traj_encoder(self):
        if not self.use_target_traj_encoder or self.target_traj_encoder is None:
            return

        tau = self.tau
        if isinstance(self.traj_encoder, WithEncoder) and isinstance(self.target_traj_encoder, WithEncoder):
            src_params = list(self.traj_encoder.module.parameters())
            tgt_params = list(self.target_traj_encoder.module.parameters())
        else:
            src_params = list(self.traj_encoder.parameters())
            tgt_params = list(self.target_traj_encoder.parameters())

        for p, tp in zip(src_params, tgt_params):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * p.data)

    def setup_logger(self, log_dir):
        self.snapshot_dir = log_dir
        setup_logger(self, log_dir)

    def log_diagnostics(self, pause_for_plot=False):
        _ = pause_for_plot
        log_diagnostics(self)

    def eval_log_diagnostics(self):
        eval_log_diagnostics(self)

    def plot_log_diagnostics(self):
        plot_log_diagnostics(self)

    def _ensure_logger(self):
        if self.logger is None or self.writer is None:
            self.setup_logger(self.snapshot_dir)

    def save(self, epoch, new_save=False, pt_save=False):
        self._ensure_logger()
        self.logger.info("Saving snapshot...")

        model_dir = os.path.join(self.snapshot_dir, f"models/epoch-{epoch}")
        if new_save and epoch != 0:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                {
                    "discrete": self.discrete,
                    "dim_skill": self.dim_skill,
                    "policy": self.sac_trainer.skill_policy,
                },
                os.path.join(model_dir, "skill_policy.pt"),
            )
            if self.stage == "pre_training":
                torch.save(
                    {
                        "discrete": self.discrete,
                        "dim_skill": self.dim_skill,
                        "traj_encoder": self.traj_encoder,
                    },
                    os.path.join(model_dir, "traj_encoder.pt"),
                )

        if pt_save and epoch != 0:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                {
                    "discrete": self.discrete,
                    "dim_skill": self.dim_skill,
                    "policy": self.sac_trainer.skill_policy,
                },
                os.path.join(model_dir, "skill_policy.pt"),
            )

        self.logger.info("Saved")

    def train(self, n_epochs):
        self.cfg.log.n_epochs = int(n_epochs)
        trainer = MetraTrainer(self.cfg, self._agent, self._env, self.replay_buffer, self.snapshot_dir)
        return trainer.train()


def sync_adapter_from_legacy(legacy_agent, adapter: LegacyMetraAdapter):
    """Copy the legacy oracle weights into the adapter for deterministic A/B tests."""

    module_pairs = [
        ("traj_encoder", "traj_encoder"),
        ("target_traj_encoder", "target_traj_encoder"),
        ("dist_predictor", "dist_predictor"),
        ("skill_dynamics", "skill_dynamics"),
    ]
    for legacy_name, adapter_name in module_pairs:
        legacy_module = getattr(legacy_agent, legacy_name, None)
        adapter_module = getattr(adapter, adapter_name, None)
        if legacy_module is None or adapter_module is None:
            continue
        adapter_module.load_state_dict(copy.deepcopy(legacy_module.state_dict()))

    adapter.sac_trainer.skill_policy.load_state_dict(copy.deepcopy(legacy_agent.sac_trainer.skill_policy.state_dict()))
    adapter.sac_trainer.qf1.load_state_dict(copy.deepcopy(legacy_agent.sac_trainer.qf1.state_dict()))
    adapter.sac_trainer.qf2.load_state_dict(copy.deepcopy(legacy_agent.sac_trainer.qf2.state_dict()))

    if hasattr(legacy_agent.sac_trainer, "target_qf1") and hasattr(adapter.sac_trainer, "target_qf1"):
        adapter.sac_trainer.target_qf1.load_state_dict(copy.deepcopy(legacy_agent.sac_trainer.target_qf1.state_dict()))
    if hasattr(legacy_agent.sac_trainer, "target_qf2") and hasattr(adapter.sac_trainer, "target_qf2"):
        adapter.sac_trainer.target_qf2.load_state_dict(copy.deepcopy(legacy_agent.sac_trainer.target_qf2.state_dict()))

    if hasattr(legacy_agent.sac_trainer, "target_qf1_mlp") and hasattr(adapter.sac_trainer, "target_qf1_mlp"):
        adapter.sac_trainer.target_qf1_mlp.load_state_dict(
            copy.deepcopy(legacy_agent.sac_trainer.target_qf1_mlp.state_dict())
        )
    if hasattr(legacy_agent.sac_trainer, "target_qf2_mlp") and hasattr(adapter.sac_trainer, "target_qf2_mlp"):
        adapter.sac_trainer.target_qf2_mlp.load_state_dict(
            copy.deepcopy(legacy_agent.sac_trainer.target_qf2_mlp.state_dict())
        )
    if hasattr(legacy_agent.sac_trainer, "target_encoder") and hasattr(adapter.sac_trainer, "target_encoder"):
        target_encoder = getattr(legacy_agent.sac_trainer, "target_encoder")
        adapter_target_encoder = getattr(adapter.sac_trainer, "target_encoder")
        if target_encoder is not None and adapter_target_encoder is not None and target_encoder is not adapter_target_encoder:
            adapter_target_encoder.load_state_dict(copy.deepcopy(target_encoder.state_dict()))

    adapter.sac_trainer.log_alpha.load_state_dict(copy.deepcopy(legacy_agent.sac_trainer.log_alpha.state_dict()))
    adapter.dual_lam.load_state_dict(copy.deepcopy(legacy_agent.dual_lam.state_dict()))
