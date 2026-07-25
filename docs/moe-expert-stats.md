# MoE expert selection statistics

`llama-server` can count how often each routed MoE expert is selected in every layer.

## Usage

Write to the default `moe-stats.csv` file:

```powershell
.\llama-server.exe `
  --model "B:\Ollama\Models\GGUF\Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf" `
  --moe-stats
```

Write to a specific file:

```powershell
.\llama-server.exe `
  --model "B:\Ollama\Models\GGUF\Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf" `
  --moe-stats-file "B:\Ollama\stats\kat-coder-moe.csv"
```

The output path can also be supplied through `LLAMA_ARG_MOE_STATS_FILE`.

Statistics are written when the server exits normally, including a normal `Ctrl+C` shutdown.

## CSV format

```csv
layer,expert,hits,layer_share
0,14,17542,0.031250000
0,37,14310,0.025489123
1,5,18443,0.032854123
```

- `layer`: transformer/MoE layer index parsed from `ffn_moe_topk`.
- `expert`: routed expert index inside that layer.
- `hits`: number of times the expert was selected.
- `layer_share`: expert selections divided by all routed selections in the same layer.

## Behavior and limitations

- Warmup is disabled automatically because warmup evaluates all experts and would contaminate the distribution.
- Shared experts are not included; only experts selected by the top-k router are counted.
- Full prompt reprocessing is counted again because it represents real repeated compute.
- The callback copies the small top-k ID tensor to CPU. Profiling therefore has some overhead and should normally be disabled for maximum throughput.
- Router mode is not supported. Start `llama-server` with a concrete model.
- Abrupt process termination can prevent the final CSV write. Stop the server with `Ctrl+C`.
