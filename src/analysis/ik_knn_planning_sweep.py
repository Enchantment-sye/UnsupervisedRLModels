from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from typing import Any

from .knn_planning import (
    DEFAULT_METHODS,
    KNNPlanningEvalConfig,
    ParsedPlanningDataset,
    QuerySet,
    _safe_nanmean,
    evaluate_single_method,
    load_or_build_nodes,
    load_or_parse_dataset,
    load_or_sample_query_bank,
    save_csv,
    slice_query_set,
)
from .maze_geodesic import dataset_slug, ensure_dir


DEFAULT_IK_ENSEMBLE_GRID = (100, 200, 400)
DEFAULT_IK_SUBSAMPLE_GRID = tuple(2**power for power in range(1, 16))
DEFAULT_IK_TEMPERATURE_GRID = (0.0001, 0.001, 0.002, 0.004, 0.008, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0)


@dataclass
class IKPlanningSweepConfig:
    datasets: list[str]
    output_dir: str
    seed: int = 0
    minari_datasets_path: str = "/home/shangyy/.minari/datasets"
    overwrite_cache: bool = False

    retrieval_top_k: int = 20
    lambda_bridge: float = 1.0
    alpha: float = 1.5
    pointmaze_h_bridge: float = 10.0
    antmaze_h_bridge: float = 15.0
    pointmaze_eps_start: float = 0.5
    pointmaze_eps_goal: float = 0.5
    antmaze_eps_start: float = 1.0
    antmaze_eps_goal: float = 1.0
    min_query_geodesic_pointmaze: float = 2.0
    min_query_geodesic_antmaze: float = 4.0
    max_query_attempts: int = 200000

    state_repr: str = "full"
    fit_pool_size: int = 50000
    gk_sigma_mode: str = "median_heuristic"
    gk_sigma: float | None = None
    mahalanobis_covariance_estimator: str = "ledoitwolf"
    mahalanobis_implementation: str = "whitening"
    mahalanobis_eps: float = 1e-6
    adaptive_k_scale: int = 10
    adaptive_eps: float = 1e-6
    ik_batch_size: int = 1024
    ik_feature_block_mb: int = 64
    ik_device: str = "auto"
    one_step_m: int = 20
    one_step_row_block_size: int = 64
    pairwise_row_block_size: int = 256

    stage1_num_queries: int = 50
    stage2_num_queries: int = 200
    stage1_base_stride_pointmaze: int = 5
    stage1_base_stride_antmaze: int = 5
    stage1_max_nodes_pointmaze_umaze: int = 1000
    stage1_max_nodes_pointmaze_large: int = 1500
    stage1_max_nodes_antmaze_umaze_diverse: int = 1000
    stage2_base_stride_pointmaze: int = 5
    stage2_base_stride_antmaze: int = 5
    stage2_max_nodes_pointmaze_umaze: int = 12000
    stage2_max_nodes_pointmaze_large: int = 15000
    stage2_max_nodes_antmaze_umaze_diverse: int = 12000

    shortlist_k: int = 5
    ik_ensemble_sizes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_IK_ENSEMBLE_GRID)
    ik_subsample_sizes: tuple[int, ...] = field(default_factory=lambda: DEFAULT_IK_SUBSAMPLE_GRID)
    ik_temperatures: tuple[float, ...] = field(default_factory=lambda: DEFAULT_IK_TEMPERATURE_GRID)
    query_bank_id: str = "ik_shared_bank_v2"
    reuse_stage2_baselines: bool = True
    max_workers: int = 1

    @property
    def shared_cache_dir(self) -> str:
        return os.path.join(self.output_dir, "cache", "shared")

    @property
    def stage1_cache_dir(self) -> str:
        return os.path.join(self.output_dir, "cache", "stage1")

    @property
    def stage2_cache_dir(self) -> str:
        return os.path.join(self.output_dir, "cache", "stage2")

    def ik_grid(self) -> list[tuple[int, int, float]]:
        return [
            (int(ensemble_size), int(subsample_size), float(temperature))
            for ensemble_size, subsample_size, temperature in product(
                self.ik_ensemble_sizes,
                self.ik_subsample_sizes,
                self.ik_temperatures,
            )
        ]


def _float_str(value: float) -> str:
    return format(float(value), ".12g")


def _ik_label(ensemble_size: int, subsample_size: int, temperature: float) -> str:
    temp_str = _float_str(temperature).replace("-", "m").replace(".", "p")
    return f"ik_e{int(ensemble_size)}_s{int(subsample_size)}_t{temp_str}"


def _load_csv_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _save_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    save_csv(path, rows, fieldnames=fieldnames)


def _numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ik_row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["dataset"]),
        int(row["ensemble_size"]),
        int(row["subsample_size"]),
        _float_str(float(row["temperature"])),
    )


def _sort_ik_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int, int, float]:
        success = _numeric(row.get("planning_success_rate"))
        path_suboptimality = _numeric(row.get("path_suboptimality"))
        precision = _numeric(row.get("precision_at_k"))
        expanded_nodes = _numeric(row.get("mean_expanded_nodes"))
        return (
            -success if math.isfinite(success) else float("inf"),
            path_suboptimality if math.isfinite(path_suboptimality) else float("inf"),
            -precision if math.isfinite(precision) else float("inf"),
            expanded_nodes if math.isfinite(expanded_nodes) else float("inf"),
            int(row["ensemble_size"]),
            int(row["subsample_size"]),
            float(row["temperature"]),
        )

    return sorted(rows, key=_selection_key)


def _augment_summary_row(
    summary: dict[str, Any],
    *,
    stage: str,
    query_count: int,
    query_bank_id: str,
    nodes_count: int,
    effective_stride: int,
    ensemble_size: int | None = None,
    subsample_size: int | None = None,
    temperature: float | None = None,
    comparison_method: str | None = None,
) -> dict[str, Any]:
    row = dict(summary)
    row["stage"] = stage
    row["query_count"] = int(query_count)
    row["query_bank_id"] = query_bank_id
    row["node_count"] = int(nodes_count)
    row["effective_stride"] = int(effective_stride)
    row["comparison_method"] = comparison_method or str(summary["method"])
    if ensemble_size is not None:
        row["ensemble_size"] = int(ensemble_size)
        row["subsample_size"] = int(subsample_size)
        row["temperature"] = float(temperature)
        row["ik_label"] = _ik_label(int(ensemble_size), int(subsample_size), float(temperature))
        row["method_family"] = "ik"
    else:
        row["ensemble_size"] = float("nan")
        row["subsample_size"] = float("nan")
        row["temperature"] = float("nan")
        row["ik_label"] = ""
        row["method_family"] = str(summary["method"])
    return row


def _build_eval_cfg(
    sweep_cfg: IKPlanningSweepConfig,
    *,
    cache_dir: str,
    cache_scope: str,
    num_queries: int,
    base_stride_pointmaze: int,
    base_stride_antmaze: int,
    max_nodes_pointmaze_umaze: int,
    max_nodes_pointmaze_large: int,
    max_nodes_antmaze_umaze_diverse: int,
) -> KNNPlanningEvalConfig:
    return KNNPlanningEvalConfig(
        datasets=list(sweep_cfg.datasets),
        output_dir=sweep_cfg.output_dir,
        cache_dir=cache_dir,
        seed=sweep_cfg.seed,
        minari_datasets_path=sweep_cfg.minari_datasets_path,
        overwrite_cache=sweep_cfg.overwrite_cache,
        base_stride_pointmaze=base_stride_pointmaze,
        base_stride_antmaze=base_stride_antmaze,
        max_nodes_pointmaze_umaze=max_nodes_pointmaze_umaze,
        max_nodes_pointmaze_large=max_nodes_pointmaze_large,
        max_nodes_antmaze_umaze_diverse=max_nodes_antmaze_umaze_diverse,
        retrieval_top_k=sweep_cfg.retrieval_top_k,
        lambda_bridge=sweep_cfg.lambda_bridge,
        alpha=sweep_cfg.alpha,
        pointmaze_h_bridge=sweep_cfg.pointmaze_h_bridge,
        antmaze_h_bridge=sweep_cfg.antmaze_h_bridge,
        pointmaze_eps_start=sweep_cfg.pointmaze_eps_start,
        pointmaze_eps_goal=sweep_cfg.pointmaze_eps_goal,
        antmaze_eps_start=sweep_cfg.antmaze_eps_start,
        antmaze_eps_goal=sweep_cfg.antmaze_eps_goal,
        num_queries=num_queries,
        min_query_geodesic_pointmaze=sweep_cfg.min_query_geodesic_pointmaze,
        min_query_geodesic_antmaze=sweep_cfg.min_query_geodesic_antmaze,
        max_query_attempts=sweep_cfg.max_query_attempts,
        state_repr=sweep_cfg.state_repr,
        fit_pool_size=sweep_cfg.fit_pool_size,
        gk_sigma_mode=sweep_cfg.gk_sigma_mode,
        gk_sigma=sweep_cfg.gk_sigma,
        mahalanobis_covariance_estimator=sweep_cfg.mahalanobis_covariance_estimator,
        mahalanobis_implementation=sweep_cfg.mahalanobis_implementation,
        mahalanobis_eps=sweep_cfg.mahalanobis_eps,
        adaptive_k_scale=sweep_cfg.adaptive_k_scale,
        adaptive_eps=sweep_cfg.adaptive_eps,
        ik_batch_size=sweep_cfg.ik_batch_size,
        ik_feature_block_mb=sweep_cfg.ik_feature_block_mb,
        ik_device=sweep_cfg.ik_device,
        one_step_m=sweep_cfg.one_step_m,
        one_step_row_block_size=sweep_cfg.one_step_row_block_size,
        pairwise_row_block_size=sweep_cfg.pairwise_row_block_size,
        plot_num_queries=1,
        cache_scope=cache_scope,
        query_bank_id=sweep_cfg.query_bank_id,
        query_bank_size=sweep_cfg.stage2_num_queries,
    )


def _stage1_cfg(sweep_cfg: IKPlanningSweepConfig) -> KNNPlanningEvalConfig:
    return _build_eval_cfg(
        sweep_cfg,
        cache_dir=sweep_cfg.stage1_cache_dir,
        cache_scope="ik_sweep_stage1",
        num_queries=sweep_cfg.stage1_num_queries,
        base_stride_pointmaze=sweep_cfg.stage1_base_stride_pointmaze,
        base_stride_antmaze=sweep_cfg.stage1_base_stride_antmaze,
        max_nodes_pointmaze_umaze=sweep_cfg.stage1_max_nodes_pointmaze_umaze,
        max_nodes_pointmaze_large=sweep_cfg.stage1_max_nodes_pointmaze_large,
        max_nodes_antmaze_umaze_diverse=sweep_cfg.stage1_max_nodes_antmaze_umaze_diverse,
    )


def _stage2_cfg(sweep_cfg: IKPlanningSweepConfig) -> KNNPlanningEvalConfig:
    return _build_eval_cfg(
        sweep_cfg,
        cache_dir=sweep_cfg.stage2_cache_dir,
        cache_scope="ik_sweep_stage2",
        num_queries=sweep_cfg.stage2_num_queries,
        base_stride_pointmaze=sweep_cfg.stage2_base_stride_pointmaze,
        base_stride_antmaze=sweep_cfg.stage2_base_stride_antmaze,
        max_nodes_pointmaze_umaze=sweep_cfg.stage2_max_nodes_pointmaze_umaze,
        max_nodes_pointmaze_large=sweep_cfg.stage2_max_nodes_pointmaze_large,
        max_nodes_antmaze_umaze_diverse=sweep_cfg.stage2_max_nodes_antmaze_umaze_diverse,
    )


def _shared_cfg(sweep_cfg: IKPlanningSweepConfig) -> KNNPlanningEvalConfig:
    return _build_eval_cfg(
        sweep_cfg,
        cache_dir=sweep_cfg.shared_cache_dir,
        cache_scope="ik_sweep_shared",
        num_queries=sweep_cfg.stage2_num_queries,
        base_stride_pointmaze=sweep_cfg.stage2_base_stride_pointmaze,
        base_stride_antmaze=sweep_cfg.stage2_base_stride_antmaze,
        max_nodes_pointmaze_umaze=sweep_cfg.stage2_max_nodes_pointmaze_umaze,
        max_nodes_pointmaze_large=sweep_cfg.stage2_max_nodes_pointmaze_large,
        max_nodes_antmaze_umaze_diverse=sweep_cfg.stage2_max_nodes_antmaze_umaze_diverse,
    )


def _final_overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["comparison_method"]), []).append(row)
    summaries = []
    for comparison_method, method_rows in sorted(grouped.items()):
        summaries.append(
            {
                "dataset": "overall",
                "comparison_method": comparison_method,
                "precision_at_k": _safe_nanmean([row["precision_at_k"] for row in method_rows]),
                "recall_at_k": _safe_nanmean([row["recall_at_k"] for row in method_rows]),
                "mean_retrieved_geodesic": _safe_nanmean([row["mean_retrieved_geodesic"] for row in method_rows]),
                "ndcg_at_k": _safe_nanmean([row["ndcg_at_k"] for row in method_rows]),
                "planning_success_rate": _safe_nanmean([row["planning_success_rate"] for row in method_rows]),
                "path_suboptimality": _safe_nanmean([row["path_suboptimality"] for row in method_rows]),
                "mean_num_retrieval_edges": _safe_nanmean([row["mean_num_retrieval_edges"] for row in method_rows]),
                "mean_path_cost": _safe_nanmean([row["mean_path_cost"] for row in method_rows]),
                "mean_expanded_nodes": _safe_nanmean([row["mean_expanded_nodes"] for row in method_rows]),
                "datasets_covered": len(method_rows),
            }
        )
    return summaries


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["| empty |", "| --- |", "| no rows |"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        formatted = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                formatted.append("nan" if not math.isfinite(value) else f"{value:.6f}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return lines


def _build_report(
    sweep_cfg: IKPlanningSweepConfig,
    stage1_shortlist_rows: list[dict[str, Any]],
    final_best_ik_rows: list[dict[str, Any]],
    final_comparison_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# IK 超参数搜索与三数据集复现实验",
        "",
        "## 1. 搜索空间",
        "",
        f"- ensemble_size: `{list(sweep_cfg.ik_ensemble_sizes)}`",
        f"- subsample_size: `{list(sweep_cfg.ik_subsample_sizes)}`",
        f"- temperature: `{list(sweep_cfg.ik_temperatures)}`",
        f"- 总 IK 配置数: `{len(sweep_cfg.ik_grid())}`",
        "",
        "## 2. 两阶段协议",
        "",
        f"- Stage 1: `{sweep_cfg.stage1_num_queries}` queries，reduced node budget。",
        f"- Stage 2: `{sweep_cfg.stage2_num_queries}` queries，full node budget。",
        f"- shortlist: 每个数据集 Top `{sweep_cfg.shortlist_k}`，按 success -> suboptimality -> precision -> expanded_nodes 排序。",
        f"- 查询集来自共享 query bank，Stage 1 使用前 `{sweep_cfg.stage1_num_queries}` 个，Stage 2 使用前 `{sweep_cfg.stage2_num_queries}` 个。",
        "",
        f"## 3. 每个数据集 Top-{sweep_cfg.shortlist_k} IK",
        "",
    ]
    for dataset in sweep_cfg.datasets:
        dataset_rows = [row for row in stage1_shortlist_rows if row.get("record_type") == "dataset_top5" and row["dataset"] == dataset]
        lines.append(f"### {dataset}")
        lines.extend(
            _markdown_table(
                dataset_rows,
                [
                    "rank",
                    "ik_label",
                    "ensemble_size",
                    "subsample_size",
                    "temperature",
                    "planning_success_rate",
                    "path_suboptimality",
                    "precision_at_k",
                    "mean_expanded_nodes",
                ],
            )
        )
        lines.append("")

    lines.extend(["## 4. 每个数据集最优 IK", ""])
    lines.extend(
        _markdown_table(
            final_best_ik_rows,
            [
                "dataset",
                "ik_label",
                "ensemble_size",
                "subsample_size",
                "temperature",
                "planning_success_rate",
                "path_suboptimality",
                "precision_at_k",
                "mean_retrieved_geodesic",
                "mean_expanded_nodes",
            ],
        )
    )
    lines.extend(["", "## 5. 最优 IK 与固定 baseline 对比", ""])
    for dataset in sweep_cfg.datasets:
        dataset_rows = [row for row in final_comparison_rows if row["dataset"] == dataset]
        lines.append(f"### {dataset}")
        lines.extend(
            _markdown_table(
                dataset_rows,
                [
                    "comparison_method",
                    "method",
                    "ik_label",
                    "precision_at_k",
                    "recall_at_k",
                    "mean_retrieved_geodesic",
                    "ndcg_at_k",
                    "planning_success_rate",
                    "path_suboptimality",
                    "mean_num_retrieval_edges",
                    "mean_path_cost",
                    "mean_expanded_nodes",
                ],
            )
        )
        lines.append("")

    lines.extend(["## 6. Overall 汇总", ""])
    lines.extend(
        _markdown_table(
            overall_rows,
            [
                "comparison_method",
                "planning_success_rate",
                "path_suboptimality",
                "precision_at_k",
                "mean_retrieved_geodesic",
                "mean_expanded_nodes",
                "datasets_covered",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 7. 说明",
            "",
            "- 最优 IK 是按每个数据集单独选择的，不强行共享一组全局超参数。",
            "- baseline 不做超参数搜索，全部按固定默认参数运行。",
            "- Stage 2 的最终对比表只展示每个数据集的最优 IK 与其余 6 个固定距离。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_ik_knn_planning_sweep(sweep_cfg: IKPlanningSweepConfig) -> dict[str, Any]:
    ensure_dir(sweep_cfg.output_dir)
    ensure_dir(sweep_cfg.shared_cache_dir)
    ensure_dir(sweep_cfg.stage1_cache_dir)
    ensure_dir(sweep_cfg.stage2_cache_dir)
    ensure_dir(os.path.join(sweep_cfg.output_dir, "tables"))
    ensure_dir(os.path.join(sweep_cfg.output_dir, "logs"))

    with open(os.path.join(sweep_cfg.output_dir, "logs", "config.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(sweep_cfg), handle, indent=2, ensure_ascii=False)

    shared_cfg = _shared_cfg(sweep_cfg)
    stage1_cfg_base = _stage1_cfg(sweep_cfg)
    stage2_cfg_base = _stage2_cfg(sweep_cfg)

    stage1_grid_path = os.path.join(sweep_cfg.output_dir, "tables", "stage1_ik_grid.csv")
    stage1_shortlist_path = os.path.join(sweep_cfg.output_dir, "tables", "stage1_ik_shortlist.csv")
    stage2_full_path = os.path.join(sweep_cfg.output_dir, "tables", "stage2_ik_full.csv")
    final_best_ik_path = os.path.join(sweep_cfg.output_dir, "tables", "final_best_ik_per_dataset.csv")
    final_comparison_per_dataset_path = os.path.join(sweep_cfg.output_dir, "tables", "final_comparison_per_dataset.csv")
    final_comparison_overall_path = os.path.join(sweep_cfg.output_dir, "tables", "final_comparison_overall.csv")
    report_path = os.path.join(sweep_cfg.output_dir, "report_ik_search_cn.md")

    parsed_datasets: dict[str, ParsedPlanningDataset] = {}
    query_banks: dict[str, QuerySet] = {}
    for dataset_id in sweep_cfg.datasets:
        parsed = load_or_parse_dataset(dataset_id, shared_cfg)
        parsed_datasets[dataset_id] = parsed
        query_banks[dataset_id] = load_or_sample_query_bank(
            parsed,
            shared_cfg,
            bank_size=sweep_cfg.stage2_num_queries,
            cache_dir=sweep_cfg.shared_cache_dir,
            query_bank_id=sweep_cfg.query_bank_id,
        )

    stage1_rows: list[dict[str, Any]] = [] if sweep_cfg.overwrite_cache else _load_csv_rows(stage1_grid_path)
    stage1_completed = {_ik_row_key(row) for row in stage1_rows}
    stage1_counter = 0
    for dataset_id in sweep_cfg.datasets:
        parsed = parsed_datasets[dataset_id]
        nodes = load_or_build_nodes(parsed, stage1_cfg_base)
        stage1_queries = slice_query_set(query_banks[dataset_id], sweep_cfg.stage1_num_queries)
        for ensemble_size, subsample_size, temperature in sweep_cfg.ik_grid():
            task_key = (dataset_id, int(ensemble_size), int(subsample_size), _float_str(float(temperature)))
            if task_key in stage1_completed and not sweep_cfg.overwrite_cache:
                continue
            eval_cfg = replace(
                stage1_cfg_base,
                ik_ensemble_size=int(ensemble_size),
                ik_subsample_size=int(subsample_size),
                ik_temperature=float(temperature),
            )
            _, merged, _, _ = evaluate_single_method(parsed, nodes, eval_cfg, "ik", stage1_queries)
            stage1_rows.append(
                _augment_summary_row(
                    merged,
                    stage="stage1",
                    query_count=stage1_queries.start_node_ids.shape[0],
                    query_bank_id=sweep_cfg.query_bank_id,
                    nodes_count=nodes.num_nodes,
                    effective_stride=nodes.effective_stride,
                    ensemble_size=ensemble_size,
                    subsample_size=subsample_size,
                    temperature=temperature,
                )
            )
            stage1_completed.add(task_key)
            stage1_counter += 1
            if stage1_counter % 10 == 0:
                _save_rows(stage1_grid_path, stage1_rows)
    _save_rows(stage1_grid_path, stage1_rows)

    shortlist_rows: list[dict[str, Any]] = []
    union_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for dataset_id in sweep_cfg.datasets:
        ranked_rows = _sort_ik_rows([row for row in stage1_rows if row["dataset"] == dataset_id])
        for rank, row in enumerate(ranked_rows[: sweep_cfg.shortlist_k], start=1):
            shortlist_row = dict(row)
            shortlist_row["rank"] = rank
            shortlist_row["record_type"] = "dataset_top5"
            shortlist_row["selected_for_stage2"] = 1
            shortlist_rows.append(shortlist_row)
            union_key = (
                int(shortlist_row["ensemble_size"]),
                int(shortlist_row["subsample_size"]),
                _float_str(float(shortlist_row["temperature"])),
            )
            existing = union_map.get(union_key)
            if existing is None:
                union_entry = {
                    "dataset": "union",
                    "record_type": "stage2_union",
                    "selected_for_stage2": 1,
                    "ensemble_size": int(shortlist_row["ensemble_size"]),
                    "subsample_size": int(shortlist_row["subsample_size"]),
                    "temperature": float(shortlist_row["temperature"]),
                    "ik_label": str(shortlist_row["ik_label"]),
                    "datasets_selected": dataset_id,
                }
                union_map[union_key] = union_entry
            else:
                datasets_selected = set(str(existing["datasets_selected"]).split(","))
                datasets_selected.add(dataset_id)
                existing["datasets_selected"] = ",".join(sorted(datasets_selected))
    shortlist_rows.extend(sorted(union_map.values(), key=lambda row: (int(row["ensemble_size"]), int(row["subsample_size"]), float(row["temperature"]))))
    _save_rows(stage1_shortlist_path, shortlist_rows)

    union_configs = [
        (int(row["ensemble_size"]), int(row["subsample_size"]), float(row["temperature"]))
        for row in shortlist_rows
        if row.get("record_type") == "stage2_union"
    ]

    stage2_ik_rows: list[dict[str, Any]] = [] if sweep_cfg.overwrite_cache else _load_csv_rows(stage2_full_path)
    stage2_completed = {_ik_row_key(row) for row in stage2_ik_rows}
    stage2_counter = 0
    for dataset_id in sweep_cfg.datasets:
        parsed = parsed_datasets[dataset_id]
        nodes = load_or_build_nodes(parsed, stage2_cfg_base)
        stage2_queries = slice_query_set(query_banks[dataset_id], sweep_cfg.stage2_num_queries)
        for ensemble_size, subsample_size, temperature in union_configs:
            task_key = (dataset_id, int(ensemble_size), int(subsample_size), _float_str(float(temperature)))
            if task_key in stage2_completed and not sweep_cfg.overwrite_cache:
                continue
            eval_cfg = replace(
                stage2_cfg_base,
                ik_ensemble_size=int(ensemble_size),
                ik_subsample_size=int(subsample_size),
                ik_temperature=float(temperature),
            )
            _, merged, _, _ = evaluate_single_method(parsed, nodes, eval_cfg, "ik", stage2_queries)
            stage2_ik_rows.append(
                _augment_summary_row(
                    merged,
                    stage="stage2",
                    query_count=stage2_queries.start_node_ids.shape[0],
                    query_bank_id=sweep_cfg.query_bank_id,
                    nodes_count=nodes.num_nodes,
                    effective_stride=nodes.effective_stride,
                    ensemble_size=ensemble_size,
                    subsample_size=subsample_size,
                    temperature=temperature,
                )
            )
            stage2_completed.add(task_key)
            stage2_counter += 1
            if stage2_counter % 5 == 0:
                _save_rows(stage2_full_path, stage2_ik_rows)
    _save_rows(stage2_full_path, stage2_ik_rows)

    best_ik_rows: list[dict[str, Any]] = []
    for dataset_id in sweep_cfg.datasets:
        dataset_stage2_rows = [row for row in stage2_ik_rows if row["dataset"] == dataset_id]
        ranked_rows = _sort_ik_rows(dataset_stage2_rows)
        if not ranked_rows:
            raise RuntimeError(f"No Stage 2 IK rows found for {dataset_id}")
        best_row = dict(ranked_rows[0])
        best_row["rank"] = 1
        best_ik_rows.append(best_row)
    _save_rows(final_best_ik_path, best_ik_rows)

    baseline_methods = [method for method in DEFAULT_METHODS if method != "ik"]
    baseline_rows: list[dict[str, Any]] = []
    existing_final_rows = [] if sweep_cfg.overwrite_cache else _load_csv_rows(final_comparison_per_dataset_path)
    existing_baseline_map = {
        (str(row["dataset"]), str(row["comparison_method"])): row
        for row in existing_final_rows
        if str(row.get("comparison_method")) not in {"best_ik_for_dataset", ""}
    }
    for dataset_id in sweep_cfg.datasets:
        parsed = parsed_datasets[dataset_id]
        nodes = load_or_build_nodes(parsed, stage2_cfg_base)
        stage2_queries = slice_query_set(query_banks[dataset_id], sweep_cfg.stage2_num_queries)
        for method in baseline_methods:
            baseline_key = (dataset_id, method)
            if sweep_cfg.reuse_stage2_baselines and baseline_key in existing_baseline_map:
                baseline_rows.append(dict(existing_baseline_map[baseline_key]))
                continue
            _, merged, _, _ = evaluate_single_method(parsed, nodes, stage2_cfg_base, method, stage2_queries)
            baseline_rows.append(
                _augment_summary_row(
                    merged,
                    stage="stage2",
                    query_count=stage2_queries.start_node_ids.shape[0],
                    query_bank_id=sweep_cfg.query_bank_id,
                    nodes_count=nodes.num_nodes,
                    effective_stride=nodes.effective_stride,
                    comparison_method=method,
                )
            )

    final_comparison_rows: list[dict[str, Any]] = []
    for best_row in best_ik_rows:
        final_row = dict(best_row)
        final_row["comparison_method"] = "best_ik_for_dataset"
        final_comparison_rows.append(final_row)
    final_comparison_rows.extend(baseline_rows)
    final_comparison_rows = sorted(final_comparison_rows, key=lambda row: (str(row["dataset"]), str(row["comparison_method"])))
    _save_rows(final_comparison_per_dataset_path, final_comparison_rows)

    overall_rows = _final_overall_summary(final_comparison_rows)
    _save_rows(final_comparison_overall_path, overall_rows)

    report = _build_report(
        sweep_cfg=sweep_cfg,
        stage1_shortlist_rows=shortlist_rows,
        final_best_ik_rows=best_ik_rows,
        final_comparison_rows=final_comparison_rows,
        overall_rows=overall_rows,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    return {
        "stage1_grid_path": stage1_grid_path,
        "stage1_shortlist_path": stage1_shortlist_path,
        "stage2_full_path": stage2_full_path,
        "final_best_ik_path": final_best_ik_path,
        "final_comparison_per_dataset_path": final_comparison_per_dataset_path,
        "final_comparison_overall_path": final_comparison_overall_path,
        "report_path": report_path,
        "shortlist_union_size": len(union_configs),
    }
