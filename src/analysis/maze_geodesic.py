from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def dataset_slug(dataset_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", dataset_id).strip("_").lower()


def _hashable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _metadata_path_from_dataset_id(
    dataset_id: str,
    minari_root: str,
) -> str:
    normalized = dataset_id.replace("D4RL/", "")
    parts = normalized.split("/")
    if len(parts) != 2:
        raise ValueError(f"Unsupported dataset id format: {dataset_id}")
    namespace, dataset_name = parts
    return os.path.join(minari_root, "D4RL", namespace, dataset_name, "data", "metadata.json")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_env_spec(metadata: dict[str, Any]) -> dict[str, Any]:
    raw_env_spec = metadata.get("env_spec")
    if raw_env_spec is None:
        raise KeyError("metadata.json does not contain env_spec")
    if isinstance(raw_env_spec, str):
        return json.loads(raw_env_spec)
    if isinstance(raw_env_spec, dict):
        return raw_env_spec
    raise TypeError(f"Unsupported env_spec type: {type(raw_env_spec)!r}")


@dataclass(frozen=True)
class MazeSpec:
    dataset_id: str
    maze_kind: str
    maze_map: np.ndarray
    maze_size_scaling: float
    free_cells: np.ndarray
    free_cell_centers: np.ndarray
    cell_shortest_paths: np.ndarray
    metadata_path: str

    @property
    def width(self) -> int:
        return int(self.maze_map.shape[1])

    @property
    def height(self) -> int:
        return int(self.maze_map.shape[0])

    def rowcol_to_xy(self, rowcol: tuple[int, int] | np.ndarray) -> np.ndarray:
        row, col = int(rowcol[0]), int(rowcol[1])
        center_x = self.maze_size_scaling * (float(col) - 0.5 * float(self.width - 1))
        center_y = self.maze_size_scaling * (float(row) - 0.5 * float(self.height - 1))
        return np.asarray([center_x, center_y], dtype=np.float32)

    def point_to_free_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(xy, dtype=np.float32)
        if points.ndim == 1:
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected (..., 2) points, got shape {points.shape}")

        tree = cKDTree(self.free_cell_centers)
        distances, indices = tree.query(points, k=1)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float32)

    def geodesic_distances(self, xy_a: np.ndarray, xy_b: np.ndarray) -> np.ndarray:
        points_a = np.asarray(xy_a, dtype=np.float32)
        points_b = np.asarray(xy_b, dtype=np.float32)
        if points_a.ndim == 1:
            points_a = points_a[None, :]
        if points_b.ndim == 1:
            points_b = points_b[None, :]
        if points_a.shape[1] != 2 or points_b.shape[1] != 2:
            raise ValueError("geodesic_distances expects 2D xy inputs")

        cell_idx_a, point_offset_a = self.point_to_free_cell(points_a)
        cell_idx_b, point_offset_b = self.point_to_free_cell(points_b)
        path_lengths = self.cell_shortest_paths[cell_idx_a[:, None], cell_idx_b[None, :]]
        distances = (
            point_offset_a[:, None].astype(np.float32)
            + np.asarray(path_lengths, dtype=np.float32)
            + point_offset_b[None, :].astype(np.float32)
        )
        return distances.astype(np.float32)

    def geodesic_for_pairs(self, xy_a: np.ndarray, xy_b: np.ndarray) -> np.ndarray:
        points_a = np.asarray(xy_a, dtype=np.float32)
        points_b = np.asarray(xy_b, dtype=np.float32)
        if points_a.shape != points_b.shape:
            raise ValueError(f"Pair inputs must share the same shape, got {points_a.shape} vs {points_b.shape}")
        if points_a.ndim == 1:
            points_a = points_a[None, :]
            points_b = points_b[None, :]
        if points_a.ndim != 2 or points_a.shape[1] != 2:
            raise ValueError("geodesic_for_pairs expects shape (N, 2)")

        cell_idx_a, point_offset_a = self.point_to_free_cell(points_a)
        cell_idx_b, point_offset_b = self.point_to_free_cell(points_b)
        cell_lengths = self.cell_shortest_paths[cell_idx_a, cell_idx_b]
        distances = point_offset_a + np.asarray(cell_lengths, dtype=np.float32) + point_offset_b
        return distances.astype(np.float32)


def build_maze_spec(
    dataset_id: str,
    maze_map: np.ndarray | list[list[Any]],
    maze_kind: str,
    *,
    maze_size_scaling: float,
    metadata_path: str = "",
    cache_dir: str | None = None,
) -> MazeSpec:
    maze_values = np.asarray(maze_map, dtype=object)
    if maze_values.ndim != 2:
        raise ValueError(f"maze_map must be 2D, got shape {maze_values.shape}")

    free_cells: list[tuple[int, int]] = []
    for row in range(maze_values.shape[0]):
        for col in range(maze_values.shape[1]):
            value = maze_values[row, col]
            if value != 1:
                free_cells.append((row, col))

    if not free_cells:
        raise ValueError("Maze contains no free cells")

    free_cell_array = np.asarray(free_cells, dtype=np.int32)
    width = int(maze_values.shape[1])
    height = int(maze_values.shape[0])
    centers = np.zeros((free_cell_array.shape[0], 2), dtype=np.float32)
    for idx, (row, col) in enumerate(free_cells):
        centers[idx, 0] = maze_size_scaling * (float(col) - 0.5 * float(width - 1))
        centers[idx, 1] = maze_size_scaling * (float(row) - 0.5 * float(height - 1))

    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_path = os.path.join(
            cache_dir,
            f"maze_geodesic_cells_{dataset_slug(dataset_id)}.npz",
        )

    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_paths = np.asarray(payload["cell_shortest_paths"], dtype=np.float32)
            cached_free_cells = np.asarray(payload["free_cells"], dtype=np.int32)
            cached_centers = np.asarray(payload["free_cell_centers"], dtype=np.float32)
            if np.array_equal(cached_free_cells, free_cell_array) and np.allclose(cached_centers, centers):
                return MazeSpec(
                    dataset_id=dataset_id,
                    maze_kind=str(maze_kind),
                    maze_map=np.asarray(maze_values),
                    maze_size_scaling=float(maze_size_scaling),
                    free_cells=free_cell_array,
                    free_cell_centers=centers,
                    cell_shortest_paths=cached_paths,
                    metadata_path=metadata_path,
                )

    rowcol_to_index = {tuple(cell.tolist()): idx for idx, cell in enumerate(free_cell_array)}
    adjacency = np.full((free_cell_array.shape[0], free_cell_array.shape[0]), np.inf, dtype=np.float32)
    np.fill_diagonal(adjacency, 0.0)
    for idx, (row, col) in enumerate(free_cells):
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neigh = (row + d_row, col + d_col)
            neigh_idx = rowcol_to_index.get(neigh)
            if neigh_idx is None:
                continue
            adjacency[idx, neigh_idx] = float(maze_size_scaling)

    graph = csr_matrix(adjacency)
    shortest_paths = shortest_path(graph, directed=False, return_predecessors=False).astype(np.float32)

    if cache_path:
        np.savez_compressed(
            cache_path,
            free_cells=free_cell_array,
            free_cell_centers=centers,
            cell_shortest_paths=shortest_paths,
        )

    return MazeSpec(
        dataset_id=dataset_id,
        maze_kind=str(maze_kind),
        maze_map=np.asarray(maze_values),
        maze_size_scaling=float(maze_size_scaling),
        free_cells=free_cell_array,
        free_cell_centers=centers,
        cell_shortest_paths=shortest_paths,
        metadata_path=metadata_path,
    )


def load_maze_spec(
    dataset_id: str,
    *,
    minari_root: str = "/home/shangyy/.minari/datasets",
    cache_dir: str | None = None,
) -> MazeSpec:
    metadata_path = _metadata_path_from_dataset_id(dataset_id, minari_root=minari_root)
    metadata = _load_json(metadata_path)
    env_spec = _resolve_env_spec(metadata)
    kwargs = dict(env_spec.get("kwargs", {}))
    maze_map = kwargs.get("maze_map")
    if maze_map is None:
        raise KeyError(f"env_spec.kwargs.maze_map missing for dataset {dataset_id}")

    lowered = dataset_id.lower()
    if "antmaze" in lowered:
        maze_kind = "antmaze"
        maze_size_scaling = float(kwargs.get("maze_size_scaling", 4.0))
    else:
        maze_kind = "pointmaze"
        maze_size_scaling = float(kwargs.get("maze_size_scaling", 1.0))

    return build_maze_spec(
        dataset_id=dataset_id,
        maze_map=maze_map,
        maze_kind=maze_kind,
        maze_size_scaling=maze_size_scaling,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
    )
