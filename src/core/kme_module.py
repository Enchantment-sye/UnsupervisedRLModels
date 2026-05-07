import torch
import numpy as np
from math import sqrt
from core.stage_contract import get_base_algo_name

class KMEModule:
    def __init__(self, config, device, kernel, replay_buffer, traj_encoder_getter, phi_from_obs_getter=None):
        self.cfg = config
        self.device = device
        self.kernel = kernel
        self.replay_buffer = replay_buffer
        self.traj_encoder_getter = traj_encoder_getter  # Function to get current (target) encoder
        self.phi_from_obs_getter = phi_from_obs_getter

        self.init_kme = False
        self.idk_step_counter = 0
        self.kme_vector = None
        self.path_datas = []  # Buffer for initialization data

    def get_kme_vector(self):
        return self.kme_vector

    def _should_record_iksd_gpu_memory(self, metrics: dict = None):
        if metrics is None:
            return False
        if self.kernel is None:
            return False
        if get_base_algo_name(self.cfg) != 'iksd':
            return False
        if not torch.cuda.is_available():
            return False
        device = self._get_cuda_device()
        return device is not None

    def _get_cuda_device(self):
        device = self.device
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != 'cuda':
            return None
        return device

    def _fit_kernel_with_gpu_memory_metrics(self, anchors, metrics: dict, metric_prefix: str):
        if not self._should_record_iksd_gpu_memory(metrics):
            self.kernel.fit(anchors)
            return

        device = self._get_cuda_device()
        torch.cuda.synchronize(device)
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        self.kernel.fit(anchors)

        torch.cuda.synchronize(device)
        after_allocated = torch.cuda.memory_allocated(device)
        after_reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        to_mb = 1024.0 ** 2

        metrics[f'{metric_prefix}_gpu_mem_peak_allocated_mb'] = peak_allocated / to_mb
        metrics[f'{metric_prefix}_gpu_mem_allocated_delta_mb'] = (after_allocated - before_allocated) / to_mb
        metrics[f'{metric_prefix}_gpu_mem_reserved_delta_mb'] = (after_reserved - before_reserved) / to_mb

    @torch.no_grad()
    def maybe_refresh_from_replay(self, metrics: dict = None):
        if self.cfg.algo.update_idk <= 0:
            return
        self.idk_step_counter += 1
        if (self.idk_step_counter % self.cfg.algo.update_idk) != 0:
            return
        anchors = self._sample_ik_state_from_replay(self.cfg.algo.idk_subsample_size)
        if anchors is None:
            if metrics is not None:
                metrics['idk_refresh_skipped'] = 1
            return
        self._fit_kernel_with_gpu_memory_metrics(anchors, metrics, 'idk_refresh')

    @torch.no_grad()
    def build_initial(self, metrics: dict = None):
        D = self.cfg.algo.dim_skill
        M = self.cfg.algo.idk_subsample_size
        
        # Force using Replay Buffer for IDK initialization
        # We skip 'gaussian'/'uniform' checks to ensure we use real data distribution
        
        anchors = None
        # Try to sample from replay buffer
        # We might need to wait or check if buffer has enough data
        # In trainer logic, this is called after some warmup, so buffer should be ready.
        
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored >= M:
             anchors = self._sample_ik_state_from_replay(M)
        
        if anchors is None:
            # If still None (e.g. buffer empty or not enough data), we must fail or warn if we want to FORCE replay buffer.
            # But for robustness, we can fall back to random but log a warning.
            # However, user asked to FORCE replay buffer.
            if self.replay_buffer is None or self.replay_buffer.n_transitions_stored == 0:
                 raise RuntimeError("IDK build_initial failed: Replay Buffer is empty! Cannot initialize IDK with data.")
            else:
                 # If just not enough for M, sample all available?
                 # Or just wait. But this function is called once.
                 # Let's try to sample as much as possible or fail if too few.
                 current_size = self.replay_buffer.n_transitions_stored
                 if current_size < M:
                      print(f"Warning: Replay buffer has {current_size} samples, less than subsample_size {M}. Using all available.")
                      anchors = self._sample_ik_state_from_replay(current_size)
                 else:
                      # Should have been sampled above
                      pass
        
        if anchors is None:
             raise RuntimeError(f"IDK build_initial failed: Could not sample anchors from Replay Buffer. Size={self.replay_buffer.n_transitions_stored if self.replay_buffer else 'None'}")

        self._fit_kernel_with_gpu_memory_metrics(anchors, metrics, 'idk_init')
        self.rebuild_mean()
        self.init_kme = True

    @torch.no_grad()
    def rebuild_mean(self):
        if self.replay_buffer is None:
            return
        kernel_embd = []
        # Sample a representative subset of the buffer to estimate mean embedding
        for _ in range(200):
            N = len(self.replay_buffer) // 200
            if N < 1: continue
            batch = self.replay_buffer.sample_transitions(N)
            all_obs = batch['obs']
            if isinstance(all_obs, np.ndarray):
                all_obs = torch.from_numpy(all_obs).float().to(self.device)
            else:
                all_obs = all_obs.to(self.device)
            all_embd = self._ik_input_from_obs(all_obs)
            kernel_embd.append(self.kernel.kernel_mean(all_embd, groups=self.cfg.algo.idk_groups))
        
        if kernel_embd:
            self.kme_vector = torch.mean(torch.stack(kernel_embd), dim=0) / sqrt(self.kernel.ensemble_size)

    @torch.no_grad()
    def _sample_ik_state_from_replay(self, n: int):
        if self.replay_buffer is None:
            return None
        if len(self.replay_buffer) < max(n, 256):
            return None
        batch = self.replay_buffer.sample_transitions(n)
        buffer_obs = batch.get('obs', None)
        if buffer_obs.dtype == np.uint8:
            obs = torch.from_numpy(buffer_obs).float().to(self.device)
        else:
            obs = torch.from_numpy(buffer_obs).float().to(self.device)
        return self._ik_input_from_obs(obs)

    @torch.no_grad()
    def _ik_input_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.phi_from_obs_getter is not None and self.cfg.algo.idk_from == 'traj':
            return self.phi_from_obs_getter(obs)
        traj_encoder = self.traj_encoder_getter()
        traj_encoder.eval()
        if self.cfg.algo.idk_from == 'traj':
            dist = traj_encoder(obs)
            phi = dist.mean
        else:
            phi = obs
        return phi

    def compute_skill_kme(self, traj_obs):
        # Used for augmenting replay buffer with skill_kme
        return np.tile(
            self.kernel.kernel_mean(self._ik_input_from_obs(traj_obs), groups=10).to('cpu').numpy() / np.sqrt(self.kernel.ensemble_size),
            (len(traj_obs), 1)
        )
