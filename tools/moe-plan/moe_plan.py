#!/usr/bin/env python3
"""Build a deterministic static MoE expert placement plan from profiler data.

The preferred path reads routed tensor metadata directly from a GGUF model, so
the resulting schema-v2 plan contains an exact tensor manifest. Legacy
placement CSV input remains supported and produces schema v1 unless it includes
the exact type/ne/nb columns emitted by the updated server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Sequence

MIB = 1024 * 1024
SCHEMA_VERSION_LEGACY = 1
SCHEMA_VERSION_MANIFEST = 2
ROUTED_TENSOR_RE = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?:gate_exps|up_exps|gate_up_exps|down_exps)\.(?:weight|bias)$"
)
FINGERPRINT_SAMPLE_BYTES = 4 * MIB
FNV1A64_OFFSET = 14695981039346656037
FNV1A64_PRIME = 1099511628211


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
    tensor_type: str = ""
    ne: tuple[int, int, int, int] | None = None
    nb: tuple[int, int, int, int] | None = None

    @property
    def expert_count(self) -> int:
        return self.expert_last - self.expert_first + 1

    @property
    def has_exact_layout(self) -> bool:
        return bool(self.tensor_type) and self.ne is not None and self.nb is not None


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


def _read_layout(row: dict[str, str], fields: set[str], source: Path, line: int):
    layout_fields = {
        "type", "ne0", "ne1", "ne2", "ne3",
        "nb0", "nb1", "nb2", "nb3",
    }
    present = layout_fields.intersection(fields)
    if not present:
        return "", None, None
    missing = layout_fields.difference(fields)
    if missing:
        raise PlanError(
            f"{source}: exact tensor layout columns are incomplete; missing {', '.join(sorted(missing))}"
        )
    tensor_type = _required(row, "type", source, line)
    ne = tuple(_parse_int(_required(row, f"ne{i}", source, line), f"ne{i}", source, line) for i in range(4))
    nb = tuple(_parse_int(_required(row, f"nb{i}", source, line), f"nb{i}", source, line) for i in range(4))
    if any(value <= 0 for value in ne) or any(value <= 0 for value in nb):
        raise PlanError(f"{source}:{line}: tensor ne/nb values must be positive")
    return tensor_type, ne, nb


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
            tensor_type, ne, nb = _read_layout(row, fields, path, line)
            if ne is not None:
                if ne[2] != count:
                    raise PlanError(f"{path}:{line}: ne2 differs from expert range")
                if nb is None or nb[2] != stride:
                    raise PlanError(f"{path}:{line}: nb2 differs from expert_stride_bytes")
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
                tensor_type=tensor_type,
                ne=ne,
                nb=nb,
            ))
    if not rows:
        raise PlanError(f"{path}: no placement rows")
    return rows


def _gguf_import():
    repo_root = Path(__file__).resolve().parents[2]
    gguf_path = repo_root / "gguf-py"
    if str(gguf_path) not in sys.path:
        sys.path.insert(0, str(gguf_path))
    try:
        from gguf import GGUFReader
        from gguf.constants import GGML_QUANT_SIZES
    except ImportError as exc:
        raise PlanError("failed to import bundled gguf-py; run from the repository checkout") from exc
    return GGUFReader, GGML_QUANT_SIZES


def _fnv1a64_update(value: int, data: bytes) -> int:
    for byte in data:
        value ^= byte
        value = (value * FNV1A64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def sampled_model_fingerprint(path: Path) -> str:
    size = path.stat().st_size
    value = _fnv1a64_update(FNV1A64_OFFSET, size.to_bytes(8, "little", signed=False))
    with path.open("rb") as handle:
        first = handle.read(min(size, FINGERPRINT_SAMPLE_BYTES))
        value = _fnv1a64_update(value, first)
        if size > FINGERPRINT_SAMPLE_BYTES:
            offset = max(FINGERPRINT_SAMPLE_BYTES, size - FINGERPRINT_SAMPLE_BYTES)
            value = _fnv1a64_update(value, offset.to_bytes(8, "little", signed=False))
            handle.seek(offset)
            value = _fnv1a64_update(value, handle.read(size - offset))
    return f"sampled-fnv1a64:{value:016x}"


def read_model_placement(path: Path) -> tuple[list[PlacementTensor], str]:
    GGUFReader, quant_sizes = _gguf_import()
    try:
        reader = GGUFReader(path)
    except (OSError, ValueError, KeyError) as exc:
        raise PlanError(f"{path}: failed to read GGUF metadata: {exc}") from exc

    rows: list[PlacementTensor] = []

    for tensor in sorted(reader.tensors, key=lambda item: item.name):
        shape_values = [int(value) for value in tensor.shape.tolist()]
        shape_values += [1] * (4 - len(shape_values))
        if len(shape_values) != 4:
            raise PlanError(f"{path}: tensor {tensor.name!r} has more than four dimensions")
        ne = tuple(shape_values)
        type_name = tensor.tensor_type.name

        match = ROUTED_TENSOR_RE.match(tensor.name)
        if match is None:
            continue

        block_size, type_size = quant_sizes[tensor.tensor_type]
        if ne[0] % block_size != 0:
            raise PlanError(
                f"{path}: tensor {tensor.name!r} ne0={ne[0]} is not divisible by quant block {block_size}"
            )
        nb0 = int(type_size)
        nb1 = nb0 * (ne[0] // int(block_size))
        nb2 = nb1 * ne[1]
        nb3 = nb2 * ne[2]
        nb = (nb0, nb1, nb2, nb3)
        computed_bytes = nb3 * ne[3]
        if computed_bytes != tensor.n_bytes:
            raise PlanError(
                f"{path}: tensor {tensor.name!r} byte size {tensor.n_bytes} "
                f"differs from computed contiguous size {computed_bytes}"
            )
        if ne[2] <= 0:
            raise PlanError(f"{path}: tensor {tensor.name!r} has no expert axis at ne2")

        rows.append(PlacementTensor(
            layer=int(match.group("layer")),
            tensor=tensor.name,
            expert_first=0,
            expert_last=ne[2] - 1,
            size_bytes=tensor.n_bytes,
            expert_stride_bytes=nb2,
            storage="GGUF",
            buffer="",
            device="",
            tensor_type=type_name,
            ne=ne,
            nb=nb,
        ))

    if not rows:
        raise PlanError(f"{path}: no supported routed MoE tensors found")
    return rows, sampled_model_fingerprint(path)


def placement_has_manifest(rows: Sequence[PlacementTensor]) -> bool:
    exact = [row.has_exact_layout for row in rows]
    if any(exact) and not all(exact):
        raise PlanError("placement metadata mixes exact and legacy tensor rows")
    return all(exact)


def tensor_manifest(rows: Sequence[PlacementTensor]) -> list[dict]:
    if not placement_has_manifest(rows):
        return []
    return [
        {
            "layer": row.layer,
            "name": row.tensor,
            "type": row.tensor_type,
            "expert_first": row.expert_first,
            "expert_last": row.expert_last,
            "size_bytes": row.size_bytes,
            "expert_stride_bytes": row.expert_stride_bytes,
            "ne": list(row.ne or ()),
            "nb": list(row.nb or ()),
        }
        for row in sorted(rows, key=lambda item: (item.layer, item.tensor))
    ]


def placement_fingerprint(rows: Sequence[PlacementTensor]) -> str:
    digest = hashlib.sha256()
    exact = placement_has_manifest(rows)
    for row in sorted(rows, key=lambda x: (x.layer, x.tensor, x.expert_first, x.expert_last)):
        normalized = (
            f"{row.layer}|{row.tensor}|{row.expert_first}|{row.expert_last}|"
            f"{row.size_bytes}|{row.expert_stride_bytes}"
        )
        if exact:
            normalized += (
                f"|{row.tensor_type}|{','.join(map(str, row.ne or ()))}|"
                f"{','.join(map(str, row.nb or ()))}"
            )
        digest.update((normalized + "\n").encode("utf-8"))
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
    source_layout: Path,
    placement_rows: Sequence[PlacementTensor],
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
    manifest = tensor_manifest(placement_rows)
    plan = {
        "schema_version": SCHEMA_VERSION_MANIFEST if manifest else SCHEMA_VERSION_LEGACY,
        "strategy": "greedy_hits_per_byte",
        "model_fingerprint": model_fingerprint,
        "placement_fingerprint": placement_hash,
        "source_profiles": [str(stats_path)],
        "source_placement": str(source_layout),
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
    if manifest:
        plan["tensor_manifest"] = manifest
    return plan


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--placement", type=Path, help="CSV emitted by --moe-placement-file")
    source.add_argument("--model", type=Path, help="GGUF model; preferred, creates exact schema-v2 manifest")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--vram-budget-mib", type=Decimal, help="routed-expert VRAM budget in MiB")
    budget.add_argument("--vram-budget-bytes", type=int, help="routed-expert VRAM budget in bytes")
    parser.add_argument("--output", type=Path, required=True, help="versioned JSON plan")
    parser.add_argument("--report", type=Path, help="optional per-expert CSV report")
    parser.add_argument("--model-fingerprint", help="optional externally computed model fingerprint override")
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="reject legacy placement metadata that cannot produce an exact tensor manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.vram_budget_bytes is not None:
            budget_bytes = args.vram_budget_bytes
        else:
            budget_bytes = int((args.vram_budget_mib * MIB).to_integral_value(rounding=ROUND_HALF_UP))
        stats = read_stats(args.stats)
        if args.model is not None:
            placement, computed_model_fingerprint = read_model_placement(args.model)
            source_layout = args.model
        else:
            placement = read_placement(args.placement)
            computed_model_fingerprint = None
            source_layout = args.placement
        if args.require_manifest and not placement_has_manifest(placement):
            raise PlanError("exact tensor manifest required; pass --model or regenerate placement CSV")
        sizes = logical_expert_sizes(placement)
        candidates = build_candidates(stats, sizes)
        selected = select_candidates(candidates, budget_bytes)
        plan = build_plan(
            candidates,
            selected,
            budget_bytes,
            args.stats,
            source_layout,
            placement,
            placement_fingerprint(placement),
            args.model_fingerprint or computed_model_fingerprint,
        )
        write_plan(args.output, plan)
        if args.report:
            write_report(args.report, candidates, selected)
        print(
            f"schema v{plan['schema_version']}: selected "
            f"{plan['selected_expert_count']}/{plan['logical_expert_count']} experts, "
            f"{plan['selected_bytes'] / MIB:.3f} MiB, "
            f"estimated GPU hit rate {plan['estimated_gpu_hit_rate'] * 100:.3f}%"
        )
        return 0
    except (OSError, PlanError, InvalidOperation) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
