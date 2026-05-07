from __future__ import annotations

import warnings
from collections import deque

import numpy as np

from envs.kitchen.metrics import KITCHEN_TASKS


class CoverageTracker:
    def __init__(self, env_name, queue_size=100000, bin_size=1.0):
        self.env_name = str(env_name or "")
        self.queue_size = int(queue_size)
        self.bin_size = float(bin_size)
        self.queue = deque(maxlen=self.queue_size)
        self.total = 0 if self._is_kitchen else set()
        self._missing_info_seen = False
        self._warned_missing_info = False

    @property
    def _is_kitchen(self):
        return "kitchen" in self.env_name.lower()

    @property
    def _coord_dims(self):
        return 1 if "cheetah" in self.env_name.lower() else 2

    def update_train_paths(self, paths):
        for path in paths or []:
            item = self._compress_path(path)
            if item is None:
                continue
            self.queue.append(item)
            if self._is_kitchen:
                self.total |= int(item)
            else:
                self.total.update(item)

    def compute_policy_metrics(self, eval_paths):
        if self._is_kitchen:
            mask, missing = self._union_kitchen_masks(eval_paths)
            metrics = {"KitchenPolicyTaskCoverage": int(mask.bit_count())}
        else:
            bins, missing = self._union_locomotion_bins(eval_paths)
            metrics = {"PolicyStateCoverageXYBins": int(len(bins))}
        if missing:
            self._missing_info_seen = True
            metrics["MissingCoverageInfo"] = 1
            self._warn_once()
        return metrics

    def compute_queue_metrics(self):
        if self._is_kitchen:
            mask = 0
            for item in self.queue:
                mask |= int(item)
            metrics = {"KitchenQueueTaskCoverage": int(mask.bit_count())}
            return self._with_missing_info(metrics)

        bins = set()
        for item in self.queue:
            bins.update(item)
        return self._with_missing_info({"QueueStateCoverageXYBins": int(len(bins))})

    def compute_total_metrics(self):
        if self._is_kitchen:
            metrics = {"KitchenTotalTaskCoverage": int(int(self.total).bit_count())}
            return self._with_missing_info(metrics)
        return self._with_missing_info({"TotalStateCoverageXYBins": int(len(self.total))})

    def state_dict(self):
        if self._is_kitchen:
            queue = [int(item) for item in self.queue]
            total = int(self.total)
        else:
            queue = [[tuple(bin_) for bin_ in item] for item in self.queue]
            total = [tuple(bin_) for bin_ in self.total]
        return {
            "env_name": self.env_name,
            "queue_size": self.queue_size,
            "bin_size": self.bin_size,
            "queue": queue,
            "total": total,
            "missing_info_seen": self._missing_info_seen,
            "warned_missing_info": self._warned_missing_info,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.env_name = str(state.get("env_name", self.env_name))
        self.queue_size = int(state.get("queue_size", self.queue_size))
        self.bin_size = float(state.get("bin_size", self.bin_size))
        self.queue = deque(maxlen=self.queue_size)

        if self._is_kitchen:
            for item in state.get("queue", []):
                self.queue.append(int(item))
            self.total = int(state.get("total", 0))
        else:
            for item in state.get("queue", []):
                self.queue.append({tuple(bin_) for bin_ in item})
            self.total = {tuple(bin_) for bin_ in state.get("total", [])}
        self._missing_info_seen = bool(state.get("missing_info_seen", False))
        self._warned_missing_info = bool(state.get("warned_missing_info", False))

    def _compress_path(self, path):
        if self._is_kitchen:
            mask, missing = self._kitchen_mask(path)
            if missing:
                self._missing_info_seen = True
                self._warn_once()
            return mask if mask is not None else None

        bins, missing = self._locomotion_bins(path)
        if missing:
            self._missing_info_seen = True
            self._warn_once()
        return bins if bins is not None else None

    def _with_missing_info(self, metrics):
        if self._missing_info_seen:
            metrics["MissingCoverageInfo"] = 1
        return metrics

    def _union_kitchen_masks(self, paths):
        mask = 0
        missing = False
        for path in paths or []:
            path_mask, path_missing = self._kitchen_mask(path)
            missing = missing or path_missing
            if path_mask is not None:
                mask |= int(path_mask)
        return mask, missing

    def _union_locomotion_bins(self, paths):
        bins = set()
        missing = False
        for path in paths or []:
            path_bins, path_missing = self._locomotion_bins(path)
            missing = missing or path_missing
            if path_bins is not None:
                bins.update(path_bins)
        return bins, missing

    def _kitchen_mask(self, path):
        env_infos = _env_infos(path)
        if not env_infos:
            return None, True

        mask = 0
        found_any = False
        for idx, (_, candidate_keys) in enumerate(KITCHEN_TASKS):
            for key in candidate_keys:
                if key not in env_infos:
                    continue
                arr = np.asarray(env_infos[key])
                if arr.size == 0:
                    continue
                found_any = True
                if bool(np.any(arr.astype(float) > 0.5)):
                    mask |= 1 << idx
                break
        return (mask, False) if found_any else (None, True)

    def _locomotion_bins(self, path):
        env_infos = _env_infos(path)
        if not env_infos:
            return None, True

        coordinates = env_infos.get("coordinates")
        next_coordinates = env_infos.get("next_coordinates")
        if coordinates is None or next_coordinates is None:
            return None, True

        coordinates = _select_coord_dims(coordinates, self._coord_dims)
        next_coordinates = _select_coord_dims(next_coordinates, self._coord_dims)
        if len(coordinates) == 0 or len(next_coordinates) == 0:
            return None, True

        coords = np.concatenate([coordinates, next_coordinates[-1:]], axis=0)
        bin_coords = np.floor(coords / self.bin_size).astype(np.int64)
        return {tuple(row.tolist()) for row in bin_coords}, False

    def _warn_once(self):
        if self._warned_missing_info:
            return
        self._warned_missing_info = True
        warnings.warn(
            "CoverageTracker missing env_infos coverage data; MissingCoverageInfo=1 will be reported.",
            RuntimeWarning,
            stacklevel=3,
        )


def _env_infos(path):
    if isinstance(path, dict):
        return path.get("env_infos", {}) or {}
    return getattr(path, "env_infos", {}) or {}


def _select_coord_dims(coords, coord_dims):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    if isinstance(coord_dims, int):
        return coords[:, :coord_dims]
    return coords[:, list(coord_dims)]
