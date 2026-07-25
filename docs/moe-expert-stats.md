# MoE expert profiling

`llama-server` can profile routed MoE expert usage and report where the routed expert tensors are stored.

## Low-overhead statistics

The default mode observes each MoE layer once per 16 evaluations. Sampling is staggered between layers, so a normal decode step usually synchronizes only a small subset of layers instead of every layer.

```powershell
.\llama-server.exe `
  --model "B:\Ollama\Models\GGUF\Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf" `
  --moe-stats-file "B:\Ollama\stats\kat-coder-moe.csv"
```

Choose a different sampling rate:

```powershell
--moe-stats-sample 32
```

`N=32` means that each layer is observed approximately once per 32 evaluations. A larger value causes less synchronization but needs a longer workload for stable results.

For exact counts, observe every routing decision:

```powershell
--moe-stats-exact
```

Exact mode is equivalent to `--moe-stats-sample 1` and can noticeably reduce throughput because it copies the top-k IDs from GPU to CPU after every MoE layer.

## Placement report

```powershell
--moe-placement-file "B:\Ollama\stats\kat-coder-placement.csv"
```

Placement is read directly from the persistent model tensors and written once immediately after the model finishes loading. It does not copy model data and does not install an eval callback, so it does not affect inference throughput.

All routed experts of one layer are stored in the same 3D tensor, so placement is reported for an expert range rather than as separate allocations for every expert ID.

## Combined example

```powershell
.\llama-server.exe `
  --model "B:\Ollama\Models\GGUF\Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf" `
  --moe-stats-file "B:\Ollama\stats\kat-coder-moe.csv" `
  --moe-stats-sample 16 `
  --moe-placement-file "B:\Ollama\stats\kat-coder-placement.csv"
```

The placement file appears after model loading. The statistics file is written during a normal shutdown, including `Ctrl+C`.

## Statistics CSV

```csv
layer,expert,hits,sampled_hits,layer_share,coverage,sample_rate
0,14,17542,1096,0.031250000,0.062500000,16
0,37,14310,894,0.025489123,0.062500000,16
```

- `hits`: estimated number of selections; exact when `sample_rate=1`.
- `sampled_hits`: selections actually copied and counted.
- `layer_share`: share of sampled selections in the layer.
- `coverage`: fraction of routing selections that were sampled.
- `sample_rate`: configured per-layer sampling interval.

The total number of router selections is known from tensor shapes even for skipped samples. This allows `hits` to be scaled using the actual sampled coverage rather than blindly multiplying by `sample_rate`.

## Placement CSV

```csv
layer,tensor,expert_first,expert_last,storage,buffer,device,size_mib
0,blk.0.ffn_gate_up_exps.weight,0,255,RAM,CPU,CPU,341.250
25,blk.25.ffn_down_exps.weight,0,255,VRAM,CUDA0,NVIDIA GeForce RTX 5070 Ti,170.625
```

Possible storage values include `RAM`, `VRAM`, `SHARED`, `ACCEL`, and `SPLIT`.

## Limitations

- Shared experts are not included; only top-k routed experts are counted.
- Reprocessed prompts are counted again because they represent real repeated compute.
- Sampling estimates become more stable over longer and more varied workloads.
- Router mode is not supported. Start `llama-server` with a concrete model.
- Abrupt process termination can prevent the final statistics file from being written.
