# llama-moe-plan (V1.1 offline planner)

This directory contains the first executable part of static per-expert placement.
It does **not** change inference. It converts existing profiler output into a
deterministic, byte-aware placement plan for the future split loader/runtime.

## Input

- `--stats`: CSV from `llama-server --moe-stats-file`.
- `--placement`: CSV from `llama-server --moe-placement-file`.
- routed-expert VRAM budget in MiB or bytes.

The new placement CSV schema includes exact `size_bytes`, `n_experts`, and
`expert_stride_bytes`. Legacy CSV files with only `size_mib` remain supported
when the reconstructed tensor size divides exactly by the expert count.

## Example

```powershell
python .\tools\moe-plan\moe_plan.py `
  --stats B:\Ollama\stats\kat-moe.csv `
  --placement B:\Ollama\stats\kat-placement.csv `
  --vram-budget-mib 10338 `
  --output B:\Ollama\stats\kat-static-plan.json `
  --report B:\Ollama\stats\kat-static-plan.csv
```

The current strategy is deterministic greedy selection by:

```text
estimated_hits / logical_expert_size_bytes
```

Missing profile rows are retained as zero-hit, low-confidence experts instead
of being treated as impossible routes.

## Test

```powershell
cd tools\moe-plan
python -m unittest -v test_moe_plan.py
```

## Next engineering step

The next commit should add a C++ plan schema/parser in core code and a dry-run
validator that loads a plan, checks model/tensor compatibility, and prints the
intended CPU/VRAM split without changing tensor allocation.
