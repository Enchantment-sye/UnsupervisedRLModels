#!/usr/bin/env python3
"""Audit METRA TensorBoard event files for official-compatible tags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

from metra_log_specs import LEGACY_ALIAS_TAGS, SUPPORTED_ENVS, get_tag_spec


EVENT_FILE_GLOB = "events.out.tfevents.*"


class AuditError(RuntimeError):
    """Raised for clear, user-facing audit failures."""


def find_event_files(logdir: str | Path) -> list[Path]:
    root = Path(logdir)
    if not root.exists():
        raise AuditError(f"logdir does not exist: {root}")
    if not root.is_dir():
        raise AuditError(f"logdir is not a directory: {root}")
    event_files = sorted(path for path in root.rglob(EVENT_FILE_GLOB) if path.is_file())
    if not event_files:
        raise AuditError(f"no TensorBoard event files found under: {root}")
    return event_files


def _load_event_file(path: Path) -> event_accumulator.EventAccumulator:
    accumulator = event_accumulator.EventAccumulator(
        str(path),
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
        },
    )
    try:
        accumulator.Reload()
    except Exception as exc:  # TensorBoard raises several parser-specific types.
        raise AuditError(f"failed to read TensorBoard event file {path}: {exc}") from exc
    return accumulator


def collect_tags(event_files: list[Path]) -> tuple[set[str], set[str], set[str], dict[str, float]]:
    scalar_tags: set[str] = set()
    histogram_tags: set[str] = set()
    image_tags: set[str] = set()
    last_scalar_by_tag: dict[str, tuple[int, float]] = {}

    for event_file in event_files:
        accumulator = _load_event_file(event_file)
        tags = accumulator.Tags()
        scalar_tags.update(tags.get("scalars", ()))
        histogram_tags.update(tags.get("histograms", ()))
        image_tags.update(tags.get("images", ()))

        for tag in tags.get("scalars", ()):
            events = accumulator.Scalars(tag)
            if not events:
                continue
            last_event = events[-1]
            current = last_scalar_by_tag.get(tag)
            if current is None or last_event.step >= current[0]:
                last_scalar_by_tag[tag] = (int(last_event.step), float(last_event.value))

    tag_last_values = {
        tag: value
        for tag, (_step, value) in sorted(last_scalar_by_tag.items())
    }
    return scalar_tags, histogram_tags, image_tags, tag_last_values


def audit_logdir(logdir: str | Path, env: str) -> dict[str, Any]:
    event_files = find_event_files(logdir)
    scalar_tags, histogram_tags, image_tags, tag_last_values = collect_tags(event_files)
    spec = get_tag_spec(env)

    all_found_tags = scalar_tags | histogram_tags | image_tags
    expected_tags = set(spec.required_tags) | set(spec.optional_tags) | set(LEGACY_ALIAS_TAGS)
    alias_hits = {
        alias: target
        for alias, target in LEGACY_ALIAS_TAGS.items()
        if alias in all_found_tags
    }

    return {
        "logdir": str(Path(logdir)),
        "env": env,
        "event_files": [str(path) for path in event_files],
        "found_scalar_tags": sorted(scalar_tags),
        "found_histogram_tags": sorted(histogram_tags),
        "found_image_tags": sorted(image_tags),
        "missing_required_tags": sorted(set(spec.required_tags) - all_found_tags),
        "missing_optional_tags": sorted(set(spec.optional_tags) - all_found_tags),
        "extra_tags": sorted(all_found_tags - expected_tags),
        "legacy_alias_tags": alias_hits,
        "tag_last_values": tag_last_values,
    }


def _format_list(values: list[str], limit: int = 12) -> str:
    if not values:
        return "-"
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(shown) + suffix


def format_human_table(result: dict[str, Any]) -> str:
    rows = [
        ("logdir", result["logdir"]),
        ("env", result["env"]),
        ("event_files", str(len(result["event_files"]))),
        ("found_scalar_tags", str(len(result["found_scalar_tags"]))),
        ("found_histogram_tags", str(len(result["found_histogram_tags"]))),
        ("found_image_tags", str(len(result["found_image_tags"]))),
        ("missing_required_tags", _format_list(result["missing_required_tags"])),
        ("missing_optional_tags", _format_list(result["missing_optional_tags"])),
        ("extra_tags", _format_list(result["extra_tags"])),
    ]
    alias_tags = result.get("legacy_alias_tags", {})
    if alias_tags:
        alias_text = "; ".join(
            f"{alias} -> {target} (alias does not satisfy required tag)"
            for alias, target in sorted(alias_tags.items())
        )
    else:
        alias_text = "-"
    rows.append(("legacy_alias_tags", alias_text))

    width = max(len(name) for name, _value in rows)
    lines = ["TensorBoard Tag Audit", "-" * 22]
    lines.extend(f"{name:<{width}} : {value}" for name, value in rows)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", required=True, help="TensorBoard log directory to audit.")
    parser.add_argument("--env", required=True, choices=SUPPORTED_ENVS)
    parser.add_argument("--strict", type=int, choices=(0, 1), default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = audit_logdir(args.logdir, args.env)
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    print()
    print(format_human_table(result))

    if args.strict and result["missing_required_tags"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
