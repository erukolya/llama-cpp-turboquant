#!/usr/bin/env python3
"""Build a deterministic static MoE expert placement plan from profiler CSV files.

The planner is intentionally offline. It does not modify model weights or inference.
It consumes the CSV files emitted by llama-server's --moe-stats and
--moe-placement options and chooses a hot expert set under a VRAM byte budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Sequence

MIB = 1024 * 1024
SCHEMA_VERSION = 1


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpertStats:
    layer: int
    expert: int
    hits: int
    sampled_hits: int
    layer_share: float
    coverage: float
    sample_rate: int


@dataclass(frozen=True)
class PlacementTensor:
    layer: int
    tensor: str
    expert_first: int
    expert_last: int
    size_bytes: int
    expert_stride_bytes: int
    storage: str
    buffer: str
    device: str

    @property
    def expert_count(self) -> int:
        return self.expert_last - self.expert_first + 1


@dataclass(frozen=True)
class ExpertCandidate:
    layer: int
    expert: int
    hits: int
    sampled_hits: int
    layer_share: float
    coverage: float
    size_bytes: int

    @property
    def score(self) -> float:
        return self.hits / self.size_bytes if self.size_bytes else 0.0


def _required(row: dict[str, str], name: str, source: Path, line: int) -> str:
    value = row.get(name)
    if value is None or value == "":
        raise PlanError(f"{source}:{line}: missing required column value '{name}'")
    return value


def _parse_int(value: str, name: str, source: Path, line: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise PlanError(f"{source}:{line}: invalid integer in '{name}': {value!r}") from exc


def _parse_float(value: str, name: str, source: Path, line: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise PlanError(f"{source}:{line}: invalid number in '{name}': {value!r}") from exc
    if not math.isfinite(result):
        raise PlanError(f"{source}:{line}: non-finite number in '{name}': {value!r}")
    return result


def read_stats(path: Path) -> list[ExpertStats]:
    rows: list[ExpertStats] = []
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "expert", "hits", "sampled_hits", "layer_share", "coverage", "sample_rate"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise PlanError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            item = ExpertStats(
                layer=_parse_int(_required(row, "layer", path, line), "layer", path, line),
                expert=_parse_int(_required(row, "expert", path, line), "expert", path, line),
                hits=_parse_int(_required(row, "hits", path, line), "hits", path, line),
                sampled_hits=_parse_int(_required(row, "sampled_hits", path, line), "sampled_hits", path, line),
                layer_share=_parse_float(_required(row, "layer_share", path, line), "layer_share", path, line),
                coverage=_parse_float(_required(row, "coverage", path, line), "coverage", path, line),
                sample_rate=_parse_int(_required(row, "sample_rate", path, line), "sample_rate", path, line),
            )
            if item.layer < 0 or item.expert < 0 or item.hits < 0 or item.sampled_hits < 0:
                raise PlanError(f"{path}:{line}: negative layer/expert/hit value")
            key = (item.layer, item.expert)
            if key in seen:
                raise PlanError(f"{path}:{line}: duplicate expert row {key}")
            seen.add(key)
            rows.append(item)
    if not rows:
        raise PlanError(f"{path}: no expert rows")
    return rows


def _mib_to_bytes(value: str, source: Path, line: int) -> int:
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:
        raise PlanError(f"{source}:{line}: invalid size_mib: {value!r}") from exc
    if decimal < 0:
        raise PlanError(f"{source}:{line}: negative size_mib")
    return int((decimal * MIB).to_integral_value(rounding=ROUND_HALF_UP))


def read_placement(path: Path) -> list[PlacementTensor]:
    rows: list[PlacementTensor] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "tensor", "expert_first", "expert_last"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise PlanError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        fields = set(reader.fieldnames or [])
        if "size_bytes" not in fields and "size_mib" not in fields:
            raise PlanError(f"{path}: requires size_bytes or size_mib")
        for line, row in enumerate(reader, start=2):
            layer = _parse_int(_required(row, "layer", path, line), "layer", path, line)
            first = _parse_int(_required(row, "expert_first", path, line), "expert_first", path, line)
            last = _parse_int(_required(row, "expert_last", path, line), "expert_last", path, line)
            if layer < 0 or first < 0 or last < first:
                raise PlanError(f"{path}:{line}: invalid layer/expert range")
            count = last - first + 1
            if row.get("size_bytes"):
                size_bytes = _parse_int(row["size_bytes"], "size_bytes", path, line)
            else:
                size_bytes = _mib_to_bytes(_required(row, "size_mib", path, line), path, line)
            if row.get("expert_stride_bytes"):
                stride = _parse_int(row["expert_stride_bytes"], "expert_stride_bytes", path, line)
            else:
                if size_bytes % count != 0:
                    raise PlanError(
                        f"{path}:{line}: tensor size {size_bytes} is not divisible by {count}; "
                        "regenerate placement CSV with expert_stride_bytes"
                    )
                stride = size_bytes // count
            if size_bytes <= 0 or stride <= 0:
                raise PlanError(f"{path}:{line}: zero or negative tensor size")
            rows.append(PlacementTensor(
                layer=layer,
                tensor=_required(row, "tensor", path, line),
                expert_first=first,
                expert_last=last,
                size_bytes=size_bytes,
                expert_stride_bytes=stride,
                storage=row.get("storage", ""),
                buffer=row.get("buffer", ""),
                device=row.get("device", ""),
            ))
    if not rows:
        raise PlanError(f"{path}: no placement rows")
    return rows


def placement_fingerprint(rows: Sequence[PlacementTensor]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda x: (x.layer, x.tensor, x.expert_first, x.expert_last)):
        normalized = (
            f"{row.layer}|{row.tensor}|{row.expert_first}|{row.expert_last}|"
            f"{row.size_bytes}|{row.expert_stride_bytes}\n"
        )
        digest.update(normalized.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def logical_expert_sizes(rows: Sequence[PlacementTensor]) -> dict[tuple[int, int], int]:
    sizes: dict[tuple[int, int], int] = defaultdict(int)
    seen_tensors: set[tuple[int, str]] = set()
    ranges_by_layer: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        tensor_key = (row.layer, row.tensor)
        if tensor_key in seen_tensors:
            raise PlanError(f"duplicate placement tensor: layer={row.layer}, tensor={row.tensor}")
        seen_tensors.add(tensor_key)
        ranges_by_layer[row.layer].add((row.expert_first, row.expert_last))
        for expert in range(row.expert_first, row.expert_last + 1):
            sizes[(row.layer, expert)] += row.expert_stride_bytes
    for layer, ranges in ranges_by_layer.items():
        if len(ranges) != 1:
            raise PlanError(f"layer {layer}: routed tensors use inconsistent expert ranges: {sorted(ranges)}")
    return dict(sizes)


def build_candidates(stats: Sequence[ExpertStats], sizes: dict[tuple[int, int], int]) -> list[ExpertCandidate]:
    stats_by_key = {(row.layer, row.expert): row for row in stats}
    unknown_stats = sorted(set(stats_by_key).difference(sizes))
    if unknown_stats:
        preview = ", ".join(map(str, unknown_stats[:5]))
        raise PlanError(f"statistics contain experts absent from placement metadata: {preview}")

    candidates: list[ExpertCandidate] = []
    for key in sorted(sizes):
        stat = stats_by_key.get(key)
        candidates.append(ExpertCandidate(
            layer=key[0],
            expert=key[1],
            hits=stat.hits if stat else 0,
            sampled_hits=stat.sampled_hits if stat else 0,
            layer_share=stat.layer_share if stat else 0.0,
            coverage=stat.coverage if stat else 0.0,
            size_bytes=sizes[key],
        ))
    return candidates


def select_candidates(candidates: Sequence[ExpertCandidate], budget_bytes: int) -> list[ExpertCandidate]:
    if budget_bytes < 0:
        raise PlanError("VRAM budget must not be negative")
    ordered = sorted(
        candidates,
        key=lambda row: (-row.score, -row.hits, row.size_bytes, row.layer, row.expert),
    )
    selected: list[ExpertCandidate] = []
    used = 0
    for row in ordered:
        if used + row.size_bytes <= budget_bytes:
            selected.append(row)
            used += row.size_bytes
    return selected


def build_plan(
    candidates: Sequence[ExpertCandidate],
    selected: Sequence[ExpertCandidate],
    budget_bytes: int,
    stats_path: Path,
    placement_path: Path,
    placement_hash: str,
    model_fingerprint: str | None,
) -> dict:
    selected_keys = {(row.layer, row.expert) for row in selected}
    total_hits = sum(row.hits for row in candidates)
    gpu_hits = sum(row.hits for row in selected)
    selected_bytes = sum(row.size_bytes for row in selected)
    layers: list[dict] = []
    by_layer: dict[int, list[ExpertCandidate]] = defaultdict(list)
    for row in candidates:
        by_layer[row.layer].append(row)
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        gpu_rows = [row for row in rows if (row.layer, row.expert) in selected_keys]
        layer_hits = sum(row.hits for row in rows)
        layer_gpu_hits = sum(row.hits for row in gpu_rows)
        layers.append({
            "layer": layer,
            "expert_count": len(rows),
            "gpu_experts": sorted(row.expert for row in gpu_rows),
            "gpu_bytes": sum(row.size_bytes for row in gpu_rows),
            "estimated_hits": layer_hits,
            "estimated_gpu_hits": layer_gpu_hits,
            "estimated_gpu_hit_rate": (layer_gpu_hits / layer_hits) if layer_hits else 0.0,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": "greedy_hits_per_byte",
        "model_fingerprint": model_fingerprint,
        "placement_fingerprint": placement_hash,
        "source_profiles": [str(stats_path)],
        "source_placement": str(placement_path),
        "vram_budget_bytes": budget_bytes,
        "selected_bytes": selected_bytes,
        "unused_budget_bytes": budget_bytes - selected_bytes,
        "logical_expert_count": len(candidates),
        "selected_expert_count": len(selected),
        "estimated_total_hits": total_hits,
        "estimated_gpu_hits": gpu_hits,
        "estimated_gpu_hit_rate": (gpu_hits / total_hits) if total_hits else 0.0,
        "layers": layers,
    }


def write_plan(path: Path, plan: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_report(path: Path, candidates: Sequence[ExpertCandidate], selected: Sequence[ExpertCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_keys = {(row.layer, row.expert) for row in selected}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "layer", "expert", "tier", "size_bytes", "estimated_hits", "sampled_hits",
            "layer_share", "coverage", "score_hits_per_byte",
        ])
        for row in sorted(candidates, key=lambda x: (x.layer, x.expert)):
            writer.writerow([
                row.layer,
                row.expert,
                "VRAM" if (row.layer, row.expert) in selected_keys else "RAM",
                row.size_bytes,
                row.hits,
                row.sampled_hits,
                f"{row.layer_share:.9f}",
                f"{row.coverage:.9f}",
                f"{row.score:.12g}",
            ])


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True, help="CSV emitted by --moe-stats-file")
    parser.add_argument("--placement", type=Path, required=True, help="CSV emitted by --moe-placement-file")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--vram-budget-mib", type=Decimal, help="routed-expert VRAM budget in MiB")
    budget.add_argument("--vram-budget-bytes", type=int, help="routed-expert VRAM budget in bytes")
    parser.add_argument("--output", type=Path, required=True, help="versioned JSON plan")
    parser.add_argument("--report", type=Path, help="optional per-expert CSV report")
    parser.add_argument("--model-fingerprint", help="optional externally computed model fingerprint")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.vram_budget_bytes is not None:
            budget_bytes = args.vram_budget_bytes
        else:
            budget_bytes = int((args.vram_budget_mib * MIB).to_integral_value(rounding=ROUND_HALF_UP))
        stats = read_stats(args.stats)
        placement = read_placement(args.placement)
        sizes = logical_expert_sizes(placement)
        candidates = build_candidates(stats, sizes)
        selected = select_candidates(candidates, budget_bytes)
        plan = build_plan(
            candidates,
            selected,
            budget_bytes,
            args.stats,
            args.placement,
            placement_fingerprint(placement),
            args.model_fingerprint,
        )
        write_plan(args.output, plan)
        if args.report:
            write_report(args.report, candidates, selected)
        print(
            f"selected {plan['selected_expert_count']}/{plan['logical_expert_count']} experts, "
            f"{plan['selected_bytes'] / MIB:.3f} MiB, "
            f"estimated GPU hit rate {plan['estimated_gpu_hit_rate'] * 100:.3f}%"
        )
        return 0
    except (OSError, PlanError, InvalidOperation) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
