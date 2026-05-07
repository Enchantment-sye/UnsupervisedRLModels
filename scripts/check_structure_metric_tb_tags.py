#!/usr/bin/env python3
"""Check structure-metric TensorBoard scalar tags in a smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_REQUIRED_TAGS = (
    "eval/Entropy_TemporalParticle_XY",
    "eval/DBI_TemporalMedoid_XY",
    "eval/Entropy_IKDE_XY",
    "eval/DBI_IKMeanRatio_Legacy_XY",
    "eval/IKSE_LegacyDBI",
    "eval/MetricBackend_UsesTemporalDistance",
    "eval/MetricBackend_UsesIsolationKernel",
    "eval/StructureMetricsNumBackends",
    "eval/StructureMetricsElapsedSec",
)

SKIP_TAGS = (
    "eval/StructureMetricsSkipped",
    "eval/StructureMetricsSkipReasonCode",
    "eval/StructureMetricsTemporalSkipped",
    "eval/StructureMetricsTemporalSkipReasonCode",
    "eval/StructureMetricsIKSkipped",
    "eval/StructureMetricsIKSkipReasonCode",
)


def _event_files(logdir: Path) -> list[Path]:
    return sorted(path for path in logdir.rglob("events.out.tfevents.*") if path.is_file())


def _load_scalars(event_file: Path) -> tuple[set[str], dict[str, float]]:
    accumulator = event_accumulator.EventAccumulator(
        str(event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", ()))
    last_values = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        if events:
            last_values[tag] = float(events[-1].value)
    return tags, last_values


def collect_scalar_tags(logdir: Path):
    event_files = _event_files(logdir)
    scalar_tags: set[str] = set()
    last_values: dict[str, float] = {}
    for event_file in event_files:
        tags, values = _load_scalars(event_file)
        scalar_tags.update(tags)
        last_values.update(values)
    return event_files, scalar_tags, last_values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True, help="Run directory or tb subdirectory.")
    parser.add_argument("--required-tag", action="append", default=None)
    args = parser.parse_args(argv)

    logdir = Path(args.logdir)
    required = tuple(args.required_tag or DEFAULT_REQUIRED_TAGS)
    event_files, scalar_tags, last_values = collect_scalar_tags(logdir)
    missing = sorted(tag for tag in required if tag not in scalar_tags)
    result = {
        "logdir": str(logdir),
        "event_files": [str(path) for path in event_files],
        "required_tags": list(required),
        "missing_tags": missing,
        "found_required_tags": sorted(set(required) & scalar_tags),
        "skip_tag_last_values": {
            tag: last_values[tag]
            for tag in SKIP_TAGS
            if tag in last_values
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not event_files:
        print(f"No TensorBoard event files found under {logdir}", file=sys.stderr)
        return 2
    if missing:
        print(f"Missing required structure metric tags: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
