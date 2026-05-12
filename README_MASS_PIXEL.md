# MASS Pixel State Coverage Training

This adds an additive `mass` training path for unsupervised state
coverage. It does not modify existing algorithm update logic. The existing
pixel SAC/DrQ-SAC actor-critic is reused as the RL control chain, while a
pre-distilled coverage encoder and NN-MASS model compute the intrinsic reward
externally.

## Method

MASS is a streaming non-parametric statistic over coverage features
`z = f_cov(o)`. A large mass means the local region has been visited often, so
the reward is small. A small mass means the region is rare, so the reward is
large.

This implementation estimates mass with nearest-neighbor Voronoi partitions:
each partition member samples `psi` anchors from the current z buffer, assigns
queries to the nearest anchor, and uses the anchor cell count for surprisal:

```text
r_i(z) = log((N + alpha * psi) / (n_i(z) + alpha))
r(z) = mean_i r_i(z)
```

Two memories are maintained:

- `B_short`: a FIFO window for the recent policy distribution.
- `B_long`: a reservoir archive for global historical coverage.

The final coverage reward is:

```text
r_cov = w_short * r_short + w_long * r_long
```

It is running mean/std normalized and clipped before action, delta-action, and
terminal penalties are applied.

## Distillation Flow

Coverage encoder distillation is algorithm-agnostic. It is launched with
`train_cov_encoder_distill.py`, not through `mass`.

1. Random environment interaction collects `(obs, action, next_obs)`.
2. A frozen local ResNet-101 teacher produces 2048-d target features.
3. `CoverageEncoder` is trained with teacher distillation, augmentation
   consistency, variance/covariance regularization, and inverse dynamics.
4. The encoder is saved as a standalone checkpoint with pixel/action/latent
   metadata.

The teacher is loaded from `/home/shangyy/models/resnet-101/` without network
access. `model.safetensors` in HuggingFace ResNet format is mapped into
`torchvision.models.resnet101(weights=None)` locally.

## MASS Training Flow

1. Random seed collection stores `(obs, action, next_obs, done, prev_action)` in
   the existing `PathBufferEx`.
2. The coverage encoder is loaded from a distillation checkpoint and frozen.
3. Seed observations are encoded to initialize short/long NN-MASS partitions.
4. Reward-free RL starts. Each critic batch recomputes intrinsic reward from the
   current MASS model before calling the existing SAC update path.

## Image Chains

- RL control chain: full pixel observation -> existing RL encoder -> actor and
  critic.
- Coverage reward chain: full pixel observation -> `CoverageEncoder` -> NN-MASS
  reward.

The MASS reward path runs under `torch.no_grad()`. It does not backpropagate to
the RL encoder or the coverage encoder. The coverage encoder is frozen so
`B_short` and `B_long` stay in a stable coordinate system.

## Commands

Coverage distillation smoke test:

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python train_cov_encoder_distill.py \
  --env debug_dummy \
  --distill_sample_steps 100 \
  --distill_train_steps 10 \
  --distill_batch_size 16 \
  --distill_lr 1e-4 \
  --cov_latent_dim 32 \
  --cov_encoder_save_path /tmp/mass_cov_encoder.pt \
  --smoke_test True
```

MASS training smoke test with a distilled checkpoint:

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python train_mass_pixel.py \
	  --env debug_dummy \
	  --algo mass \
	  --encoder 1 \
	  --cov_encoder_type checkpoint \
  --cov_encoder_path /tmp/mass_cov_encoder.pt \
  --seed_steps 100 \
  --n_epochs 2 \
  --traj_batch_size 1 \
  --trans_optimization_epochs 10 \
  --trans_minibatch_size 16 \
  --sac_min_buffer_size 16 \
  --sac_max_buffer_size 500 \
  --n_parallel 2 \
  --parallel_sampler_num_workers 2 \
  --parallel_sampler_enabled \
  --n_epochs_per_eval 1 \
  --num_random_trajectories 9 \
  --eval_state_coverage_trajectories 48 \
  --eval_record_video True \
  --mass_c 4 \
  --mass_psi 8 \
	  --mass_short_size 200 \
	  --mass_long_size 500 \
	  --mass_encode_batch_size 64 \
	  --mass_refresh_interval 1 \
	  --mass_refresh_num 1 \
  --smoke_test True
```

MASS training smoke test with direct ResNet-101 features, no distillation:

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python train_mass_pixel.py \
	  --env debug_dummy \
	  --algo mass \
	  --encoder 1 \
	  --cov_encoder_type resnet-101 \
  --cov_resnet_path /home/shangyy/models/resnet-101/ \
  --seed_steps 100 \
  --n_epochs 2 \
  --traj_batch_size 1 \
  --trans_optimization_epochs 10 \
  --trans_minibatch_size 16 \
  --sac_min_buffer_size 16 \
  --sac_max_buffer_size 500 \
  --n_parallel 2 \
  --parallel_sampler_num_workers 2 \
  --parallel_sampler_enabled \
  --n_epochs_per_eval 1 \
  --num_random_trajectories 9 \
  --eval_state_coverage_trajectories 48 \
  --eval_record_video True \
  --mass_c 4 \
  --mass_psi 8 \
	  --mass_short_size 200 \
	  --mass_long_size 500 \
	  --mass_encode_batch_size 64 \
	  --mass_refresh_interval 1 \
	  --mass_refresh_num 1 \
  --smoke_test True
```

MASS training smoke test with direct DINOv3 features, no distillation:

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python train_mass_pixel.py \
	  --env debug_dummy \
	  --algo mass \
	  --encoder 1 \
	  --cov_encoder_type dinov3 \
  --cov_dino_path /home/shangyy/models/dinov3-vits16-pretrain-lvd1689m/ \
  --seed_steps 100 \
  --n_epochs 2 \
  --traj_batch_size 1 \
  --trans_optimization_epochs 10 \
  --trans_minibatch_size 16 \
  --sac_min_buffer_size 16 \
  --sac_max_buffer_size 500 \
  --n_parallel 2 \
  --parallel_sampler_num_workers 2 \
  --parallel_sampler_enabled \
  --n_epochs_per_eval 1 \
  --num_random_trajectories 9 \
  --eval_state_coverage_trajectories 48 \
  --eval_record_video True \
  --mass_c 4 \
  --mass_psi 8 \
	  --mass_short_size 200 \
	  --mass_long_size 500 \
	  --mass_encode_batch_size 64 \
	  --mass_refresh_interval 1 \
	  --mass_refresh_num 1 \
  --smoke_test True
```

Full run example:

```bash
/home/shangyy/miniconda3/envs/metra_idk/bin/python train_cov_encoder_distill.py \
  --env dmc_walker_walk \
  --distill_sample_steps 5000 \
  --distill_train_steps 2000 \
  --distill_batch_size 256 \
  --distill_lr 1e-4 \
  --cov_encoder_save_path /tmp/walker_cov_encoder.pt

/home/shangyy/miniconda3/envs/metra_idk/bin/python train_mass_pixel.py \
	  --env dmc_walker_walk \
	  --algo mass \
	  --encoder 1 \
	  --cov_encoder_type checkpoint \
  --cov_encoder_path /tmp/walker_cov_encoder.pt \
  --seed_steps 5000 \
  --n_epochs 1000000 \
  --traj_batch_size 8 \
  --trans_optimization_epochs 200 \
  --trans_minibatch_size 256 \
  --sac_min_buffer_size 10000 \
  --sac_max_buffer_size 300000
```

## Notes

- `mass` supports `--cov_encoder_type checkpoint`, `resnet-101`, or
  `dinov3` / `dino-v3`. The direct ResNet/DINO paths load local frozen vision
  models and do not run distillation.
- MASS eval is intentionally skill-discovery-free: it logs mean return/length,
  state coverage over `--eval_state_coverage_trajectories` deterministic
  rollouts, and, when `--eval_record_video True`, records
  `--num_random_trajectories` policy rollouts as a 3x3 video grid under
  `eval/random_rollouts_3x3` and `videos/eval_epoch-*_3x3.mp4`.
- `--eval_plot_axis`, `--alpha`, `--sac_lr_a`, and
  `--replay_staging_enabled` follow the same CLI meaning as the existing
  skill-discovery entrypoint. `--alpha` is SAC temperature; NN-MASS smoothing is
  `--mass_alpha`.
- `--mass_encode_batch_size` controls the micro-batch size for
  `obs -> coverage feature z` encoding. Lower it for DINOv3 OOMs; raise it when
  memory is available.
- `--mass_device auto` keeps MASS buffers/counts on the training device for
  speed. Use `--mass_device cpu` only when z buffers/counts create GPU memory
  pressure.
- MASS rollout supports the repo's generic process-parallel sampler for
  standard URL-style envs. Use `--n_parallel`, `--parallel_sampler_num_workers`,
  `--parallel_sampler_enabled`, and `--eval_parallel_sampler_enabled`; unsupported
  specialized env families fall back to serial rollout.
- `--mass_refresh_interval` is measured in training epochs. The default is 5;
  set it to 1 for smoke tests or to 0 to disable partition refresh.
- `online_cov_update` is intentionally disabled in v1. If it is enabled later,
  `B_short` and `B_long` must be re-encoded when partitions refresh.
- Existing algorithm, agent, replay, encoder, logger, and global config files
  are left untouched.
