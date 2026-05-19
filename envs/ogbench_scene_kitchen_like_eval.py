from __future__ import annotations

import math

import numpy as np


DRAWER_OPEN_THRESH = -0.12
DRAWER_CLOSED_THRESH = 0.04
WINDOW_OPEN_THRESH = 0.15
WINDOW_CLOSED_THRESH = 0.04
CUBE_MOVE_THRESH = 0.08
CUBE_LIFT_Z_THRESH = 0.06
DRAWER_CUBE_TARGET = np.array([0.33, -0.356, 0.066], dtype=np.float64)
CUBE_IN_DRAWER_THRESH = 0.07

METRIC_PREFIX = "OGBenchSceneKitchen"
RESET_INFO_PREFIX = "ogbench_scene/reset/"

PREDICATES = (
    "button0_toggled",
    "button1_toggled",
    "drawer_opened",
    "drawer_closed",
    "window_opened",
    "window_closed",
    "cube_moved",
    "cube_lifted",
    "cube_to_table_region",
    "cube_in_drawer",
)

_CUBE_POS_KEY = "privileged/block_0_pos"
_BUTTON_STATES_KEY = "button_states"
_BUTTON0_STATE_KEY = "privileged/button_0_state"
_BUTTON1_STATE_KEY = "privileged/button_1_state"
_DRAWER_POS_KEY = "privileged/drawer_pos"
_WINDOW_POS_KEY = "privileged/window_pos"

SCENE_STEP_REQUIRED_KEYS = (
    _CUBE_POS_KEY,
    _DRAWER_POS_KEY,
    _WINDOW_POS_KEY,
)

SCENE_RESET_INFO_KEYS = (
    _CUBE_POS_KEY,
    _BUTTON_STATES_KEY,
    _BUTTON0_STATE_KEY,
    _BUTTON1_STATE_KEY,
    _DRAWER_POS_KEY,
    _WINDOW_POS_KEY,
)


def is_ogbench_scene_metric_key(key: str) -> bool:
    return isinstance(key, str) and key.startswith(f"{METRIC_PREFIX}/")


def is_ogbench_scene_info_dict(info: dict) -> bool:
    if not isinstance(info, dict):
        return False
    if all(key in info for key in SCENE_STEP_REQUIRED_KEYS):
        return _has_button_state(info)
    return False


def reset_info_key(key: str) -> str:
    return f"{RESET_INFO_PREFIX}{key}"


def calc_ogbench_scene_kitchen_like_metrics(trajectories) -> dict:
    if not _trajectories_look_like_ogbench_scene(trajectories):
        return {}

    results = []
    for traj_idx, traj in enumerate(trajectories or []):
        env_infos = _trajectory_env_infos(traj)
        if not env_infos:
            raise KeyError(
                "Missing env_infos for OGBench Scene metrics. The rollout "
                "collector may be dropping step info."
            )

        reset_info = _extract_reset_info_from_env_infos(env_infos, traj_idx=traj_idx)
        num_steps = _infer_num_steps(env_infos)
        evaluator = OGBenchSceneKitchenLikeEvaluator()
        evaluator.reset(reset_info)
        for step_idx in range(num_steps):
            evaluator.update(_extract_step_info(env_infos, step_idx))
        results.append(evaluator.result())

    return OGBenchSceneKitchenLikeEvaluator.aggregate(results)


class OGBenchSceneKitchenLikeEvaluator:
    """Evaluation-only privileged-state metrics for OGBench Scene.

    This evaluator must only be used on evaluation rollouts. It reads
    privileged object states from OGBench ``info`` dictionaries and never
    changes rewards, replay data, policy inputs, or optimizer updates.
    """

    def __init__(
        self,
        xy_bin_size=0.05,
        drawer_bin_size=0.04,
        window_bin_size=0.04,
        cube_table_region_y=0.12,
    ):
        self.xy_bin_size = float(xy_bin_size)
        self.drawer_bin_size = float(drawer_bin_size)
        self.window_bin_size = float(window_bin_size)
        self.cube_table_region_y = float(cube_table_region_y)
        self._is_reset = False
        self.reset_info = None

    def reset(self, reset_info: dict) -> None:
        if not isinstance(reset_info, dict):
            raise TypeError(
                "OGBench Scene evaluator reset_info must be a dict. "
                "Check that env.last_reset_info is preserved by the adapter."
            )

        init_cube_pos = _extract_cube_pos(reset_info)
        init_button_states = _extract_button_states(reset_info)
        init_drawer_pos = _extract_scalar(reset_info, _DRAWER_POS_KEY)
        init_window_pos = _extract_scalar(reset_info, _WINDOW_POS_KEY)

        self.reset_info = dict(reset_info)
        self.init_cube_pos = init_cube_pos
        self.init_cube_xy = init_cube_pos[:2].copy()
        self.init_button_states = init_button_states
        self.init_drawer_pos = init_drawer_pos
        self.init_window_pos = init_window_pos

        self.predicates = {name: False for name in PREDICATES}
        self.events = {}
        self.step_idx = 0
        self._saw_drawer_opened = False
        self._saw_window_opened = False

        self.visited_cube_xy_bins = set()
        self.visited_drawer_position_bins = set()
        self.visited_window_position_bins = set()
        self.visited_button_states = set()

        self.max_cube_xy_displacement = 0.0
        self.max_cube_z = float(init_cube_pos[2])
        self.min_drawer_pos = float(init_drawer_pos)
        self.max_window_pos = float(init_window_pos)

        self._record_object_coverage(
            init_cube_pos,
            init_button_states,
            init_drawer_pos,
            init_window_pos,
        )
        self._is_reset = True

    def update(self, info: dict) -> None:
        if not self._is_reset:
            raise RuntimeError("OGBenchSceneKitchenLikeEvaluator.update() called before reset().")
        if not isinstance(info, dict):
            raise TypeError("OGBench Scene step info must be a dict.")

        cube_pos = _extract_cube_pos(info)
        button_states = _extract_button_states(info)
        drawer_pos = _extract_scalar(info, _DRAWER_POS_KEY)
        window_pos = _extract_scalar(info, _WINDOW_POS_KEY)

        self._record_object_coverage(cube_pos, button_states, drawer_pos, window_pos)
        cube_xy_disp = float(np.linalg.norm(cube_pos[:2] - self.init_cube_xy))
        self.max_cube_xy_displacement = max(self.max_cube_xy_displacement, cube_xy_disp)
        self.max_cube_z = max(self.max_cube_z, float(cube_pos[2]))
        self.min_drawer_pos = min(self.min_drawer_pos, float(drawer_pos))
        self.max_window_pos = max(self.max_window_pos, float(window_pos))

        self._maybe_mark("button0_toggled", button_states[0] != self.init_button_states[0])
        self._maybe_mark("button1_toggled", button_states[1] != self.init_button_states[1])

        drawer_open = drawer_pos <= DRAWER_OPEN_THRESH
        self._maybe_mark("drawer_opened", drawer_open)
        if drawer_open:
            self._saw_drawer_opened = True
        self._maybe_mark(
            "drawer_closed",
            self._saw_drawer_opened and abs(drawer_pos) <= DRAWER_CLOSED_THRESH,
        )

        window_open = window_pos >= WINDOW_OPEN_THRESH
        self._maybe_mark("window_opened", window_open)
        if window_open:
            self._saw_window_opened = True
        self._maybe_mark(
            "window_closed",
            self._saw_window_opened and abs(window_pos) <= WINDOW_CLOSED_THRESH,
        )

        self._maybe_mark("cube_moved", cube_xy_disp >= CUBE_MOVE_THRESH)
        self._maybe_mark("cube_lifted", float(cube_pos[2]) >= CUBE_LIFT_Z_THRESH)
        self._maybe_mark("cube_to_table_region", float(cube_pos[1]) >= self.cube_table_region_y)
        self._maybe_mark(
            "cube_in_drawer",
            float(np.linalg.norm(cube_pos - DRAWER_CUBE_TARGET)) <= CUBE_IN_DRAWER_THRESH,
        )
        self.step_idx += 1

    def result(self) -> dict:
        return {
            "predicates": dict(self.predicates),
            "events": dict(self.events),
            "visited_cube_xy_bins": set(self.visited_cube_xy_bins),
            "visited_drawer_position_bins": set(self.visited_drawer_position_bins),
            "visited_window_position_bins": set(self.visited_window_position_bins),
            "visited_button_states": set(self.visited_button_states),
            "max_cube_xy_displacement": float(self.max_cube_xy_displacement),
            "max_cube_z": float(self.max_cube_z),
            "min_drawer_pos": float(self.min_drawer_pos),
            "max_window_pos": float(self.max_window_pos),
        }

    @staticmethod
    def aggregate(traj_results: list[dict]) -> dict:
        traj_results = list(traj_results or [])
        num_trajs = len(traj_results)
        predicate_matrix = np.zeros((num_trajs, len(PREDICATES)), dtype=bool)

        for traj_idx, result in enumerate(traj_results):
            predicates = result.get("predicates", {}) or {}
            for pred_idx, name in enumerate(PREDICATES):
                predicate_matrix[traj_idx, pred_idx] = bool(predicates.get(name, False))

        predicate_union = predicate_matrix.any(axis=0) if num_trajs else np.zeros(len(PREDICATES), dtype=bool)
        completed_per_traj = predicate_matrix.sum(axis=1) if num_trajs else np.asarray([], dtype=np.int64)

        metrics = {
            f"{METRIC_PREFIX}/AtomicCoverage": int(predicate_union.sum()),
            f"{METRIC_PREFIX}/AtomicCoverageRatio": float(predicate_union.mean()) if len(PREDICATES) else 0.0,
            f"{METRIC_PREFIX}/MeanAtomicCompletion": float(completed_per_traj.mean()) if num_trajs else 0.0,
            f"{METRIC_PREFIX}/MaxAtomicCompletion": int(completed_per_traj.max()) if num_trajs else 0,
        }

        for pred_idx, name in enumerate(PREDICATES):
            metrics[f"{METRIC_PREFIX}/Predicate/{name}"] = (
                float(predicate_matrix[:, pred_idx].mean()) if num_trajs else 0.0
            )
            metrics[f"{METRIC_PREFIX}/PredicateUnion/{name}"] = int(predicate_union[pred_idx])

        cube_bins = _union_sets(traj_results, "visited_cube_xy_bins")
        drawer_bins = _union_sets(traj_results, "visited_drawer_position_bins")
        window_bins = _union_sets(traj_results, "visited_window_position_bins")
        button_states = _union_sets(traj_results, "visited_button_states")

        metrics.update({
            f"{METRIC_PREFIX}/ObjectCoverage/CubeXYBinCoverage": int(len(cube_bins)),
            f"{METRIC_PREFIX}/ObjectCoverage/DrawerPositionCoverage": int(len(drawer_bins)),
            f"{METRIC_PREFIX}/ObjectCoverage/WindowPositionCoverage": int(len(window_bins)),
            f"{METRIC_PREFIX}/ObjectCoverage/ButtonStateCoverage": int(len(button_states)),
            f"{METRIC_PREFIX}/ObjectCoverage/ButtonStateCoverageRatio": float(len(button_states) / 4.0),
            f"{METRIC_PREFIX}/ObjectCoverage/CubeInDrawerRate": metrics[f"{METRIC_PREFIX}/Predicate/cube_in_drawer"],
            f"{METRIC_PREFIX}/ObjectCoverage/DrawerOpenRate": metrics[f"{METRIC_PREFIX}/Predicate/drawer_opened"],
            f"{METRIC_PREFIX}/ObjectCoverage/WindowOpenRate": metrics[f"{METRIC_PREFIX}/Predicate/window_opened"],
            f"{METRIC_PREFIX}/ObjectCoverage/MeanMaxCubeXYDisplacement": _mean_result_value(
                traj_results, "max_cube_xy_displacement"
            ),
            f"{METRIC_PREFIX}/ObjectCoverage/MaxCubeXYDisplacement": _max_result_value(
                traj_results, "max_cube_xy_displacement"
            ),
            f"{METRIC_PREFIX}/ObjectCoverage/MeanMaxCubeZ": _mean_result_value(traj_results, "max_cube_z"),
            f"{METRIC_PREFIX}/ObjectCoverage/MeanDrawerMinPos": _mean_result_value(traj_results, "min_drawer_pos"),
            f"{METRIC_PREFIX}/ObjectCoverage/MeanWindowMaxPos": _mean_result_value(traj_results, "max_window_pos"),
        })
        return metrics

    def _maybe_mark(self, name: str, condition: bool) -> None:
        if condition and not self.predicates[name]:
            self.predicates[name] = True
            self.events[name] = int(self.step_idx)

    def _record_object_coverage(self, cube_pos, button_states, drawer_pos, window_pos) -> None:
        self.visited_cube_xy_bins.add((
            _bin_value(cube_pos[0], self.xy_bin_size),
            _bin_value(cube_pos[1], self.xy_bin_size),
        ))
        self.visited_drawer_position_bins.add(_bin_value(drawer_pos, self.drawer_bin_size))
        self.visited_window_position_bins.add(_bin_value(window_pos, self.window_bin_size))
        self.visited_button_states.add((int(button_states[0]), int(button_states[1])))


def _trajectories_look_like_ogbench_scene(trajectories) -> bool:
    for traj in trajectories or []:
        env_infos = _trajectory_env_infos(traj)
        if not env_infos:
            continue
        keys = set(env_infos.keys())
        if any(key.startswith(RESET_INFO_PREFIX) for key in keys):
            return True
        if all(key in keys for key in SCENE_STEP_REQUIRED_KEYS) and (
            _BUTTON_STATES_KEY in keys
            or (_BUTTON0_STATE_KEY in keys and _BUTTON1_STATE_KEY in keys)
        ):
            return True
    return False


def _extract_reset_info_from_env_infos(env_infos: dict, *, traj_idx: int) -> dict:
    reset_info = {}
    for key in SCENE_RESET_INFO_KEYS:
        prefixed_key = reset_info_key(key)
        if prefixed_key not in env_infos:
            continue
        values = _as_array(env_infos[prefixed_key])
        if len(values) == 0:
            continue
        reset_info[key] = values[0]

    try:
        _extract_cube_pos(reset_info)
        _extract_button_states(reset_info)
        _extract_scalar(reset_info, _DRAWER_POS_KEY)
        _extract_scalar(reset_info, _WINDOW_POS_KEY)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Missing OGBench Scene reset info {missing!r} for trajectory {traj_idx}. "
            "Ensure the OGBench adapter preserves env.last_reset_info and injects "
            "reset fields into evaluation step info."
        ) from exc
    return reset_info


def _extract_step_info(env_infos: dict, step_idx: int) -> dict:
    info = {}
    for key, values in env_infos.items():
        if key.startswith(RESET_INFO_PREFIX):
            continue
        arr = _as_array(values)
        if arr.ndim == 0:
            continue
        if arr.shape[0] <= step_idx:
            continue
        info[key] = arr[step_idx]
    return info


def _infer_num_steps(env_infos: dict) -> int:
    lengths = []
    for key, values in env_infos.items():
        if key.startswith(RESET_INFO_PREFIX):
            continue
        try:
            lengths.append(len(values))
        except TypeError:
            pass
    if not lengths:
        raise KeyError(
            "No per-step OGBench Scene info found. The adapter or rollout "
            "collector may be dropping step info."
        )
    return int(max(lengths))


def _trajectory_env_infos(traj):
    if isinstance(traj, dict):
        return traj.get("env_infos", {}) or {}
    return getattr(traj, "env_infos", {}) or {}


def _extract_cube_pos(info: dict) -> np.ndarray:
    value = _require_key(info, _CUBE_POS_KEY)
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError(
            f"OGBench Scene field {_CUBE_POS_KEY!r} must contain xyz position, got shape={arr.shape}."
        )
    return arr[:3].astype(np.float64, copy=True)


def _extract_button_states(info: dict) -> np.ndarray:
    if _BUTTON_STATES_KEY in info:
        arr = np.asarray(info[_BUTTON_STATES_KEY]).squeeze()
    else:
        button0 = _require_key(info, _BUTTON0_STATE_KEY)
        button1 = _require_key(info, _BUTTON1_STATE_KEY)
        arr = np.asarray([_extract_button_scalar(button0), _extract_button_scalar(button1)])
    arr = np.asarray(arr).reshape(-1)
    if arr.size < 2:
        raise ValueError(
            "OGBench Scene button state must contain two entries via "
            f"{_BUTTON_STATES_KEY!r} or the privileged button fallback keys."
        )
    return np.asarray(arr[:2], dtype=np.int64)


def _extract_button_scalar(value) -> int:
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        raise ValueError(
            "OGBench Scene fallback button state must be scalar or shape=(1,), "
            f"got shape={arr.shape}."
        )
    return int(arr.reshape(-1)[0])


def _extract_scalar(info: dict, key: str) -> float:
    value = _require_key(info, key)
    arr = np.asarray(value, dtype=np.float64).squeeze()
    if arr.size != 1:
        raise ValueError(f"OGBench Scene field {key!r} must be scalar or shape=(1,), got shape={arr.shape}.")
    return float(arr.reshape(-1)[0])


def _require_key(info: dict, key: str):
    if key not in info:
        available = ", ".join(sorted(str(candidate) for candidate in info.keys()))
        raise KeyError(
            f"Missing OGBench Scene info field {key!r}. This metric requires "
            "OGBench Scene privileged info; check that the env is an OGBench "
            "Scene env and the adapter preserves reset/step info. "
            f"Available keys: [{available}]"
        )
    return info[key]


def _has_button_state(info: dict) -> bool:
    return _BUTTON_STATES_KEY in info or (_BUTTON0_STATE_KEY in info and _BUTTON1_STATE_KEY in info)


def _bin_value(value, bin_size: float) -> int:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    return int(math.floor(float(value) / float(bin_size) + 1e-12))


def _union_sets(results, key: str) -> set:
    union = set()
    for result in results:
        union.update(result.get(key, set()) or set())
    return union


def _mean_result_value(results, key: str) -> float:
    if not results:
        return 0.0
    return float(np.mean([float(result.get(key, 0.0)) for result in results]))


def _max_result_value(results, key: str) -> float:
    if not results:
        return 0.0
    return float(np.max([float(result.get(key, 0.0)) for result in results]))


def _is_ragged(values) -> bool:
    try:
        np.asarray(values)
    except ValueError:
        return True
    return False


def _as_array(values):
    return np.asarray(values, dtype=object) if _is_ragged(values) else np.asarray(values)
