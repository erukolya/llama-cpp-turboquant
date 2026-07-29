# User validation policy for static MoE placement

This document defines when work may continue without user involvement and when validation on the primary Windows/CUDA/KAT machine is required.

## Communication rule

The implementation agent must use one of these explicit statuses:

```text
USER NOT NEEDED
```

Work can continue using source inspection, unit tests, CI, synthetic models, static validation or repository artifacts.

```text
USER CHECK REQUIRED
```

A result depends on the user's local KAT GGUF, RTX 5070 Ti, RAM/VRAM measurements, PCIe behavior or real workload.

When a user check is required, provide:

1. one exact branch or artifact;
2. one ready-to-run command per required model/quant;
3. expected success markers;
4. exact output files to return;
5. no unrelated manual setup.

Do not ask the user to inspect source code, build partial commits manually or repeat checks already covered by CI.

## Checks performed without the user

The agent should handle these independently:

- source and architecture analysis;
- planner implementation;
- JSON schema and parser tests;
- CLI parsing tests;
- CPU and Windows builds;
- server build checks;
- malformed-plan tests;
- synthetic tensor-layout validation;
- backend route-map and missing-slot execution tests;
- static memory accounting;
- documentation and progress tracking;
- CI failure diagnosis and fixes.

## Hardware checkpoints

The target is no more than three user checkpoints for Version 1.

### U1 — KAT model-aware dry run

Completed for Q5_K_M and Q4_K_M.

The check validated every routed tensor name, type, `ne[]`, `nb[]`, expert stride, model fingerprint and intended CPU/VRAM split without changing allocation or inference behavior.

### U2 — Exclusive split-loader memory validation

Accepted for Q5_K_M. The Q4_K_M upload structurally confirmed compact CPU/CUDA pools and zero packed runtime tensors, but used the Q5 plan; final U3 repeats exact accounting with the matching Q4 plan.

The check reports:

- exact planned and actual CPU/GPU routed bytes;
- logical and allocated pool bytes;
- compact tensor count;
- `packed_runtime_tensors=0`;
- successful unload.

### U3 — Final correctness, PCIe and execution validation

Trigger only after:

- mixed CPU/CUDA execution is implemented;
- host and real backend route tests pass on Windows and Ubuntu;
- CPU/accelerator execution counters are exposed;
- routed-weight copy counters are exposed;
- real `llama-server --moe-expert-plan` inference is enabled;
- the Windows CUDA 13.3 artifact is available.

The U3 runner performs in one command:

1. exact U2 compact-pool accounting;
2. deterministic stock baseline generation;
3. deterministic static-placement generation;
4. exact output comparison;
5. TurboQuant KV validation with `K=turbo4`, `V=turbo3` and flash attention;
6. CPU `MUL_MAT_ID` execution count > 0;
7. accelerator `MUL_MAT_ID` execution count > 0;
8. all routed-weight copy counters exactly zero;
9. RAM, VRAM, server arguments, responses, metrics and startup/shutdown logs;
10. automatic ZIP creation on both success and failure.

## U3 artifact

Artifact name:

```text
moe-u3-windows-x64-cuda13.3-sm120
```

Extract it into a new directory. The `.cmd` wrapper invokes PowerShell with `ExecutionPolicy Bypass`, so no signing or `Unblock-File` step is required.

## U3 Q4_K_M command

Run from the extracted artifact directory:

```powershell
.\run-moe-u3.cmd `
  -Model "B:\Ollama\Models\GGUF\Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf" `
  -Plan "B:\Ollama\moe-u1-q4\kat-static-plan.json"
```

Expected output ZIP:

```text
moe-u3-result-Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.zip
```

## U3 Q5_K_M command

Run from the same extracted artifact directory after Q4 finishes:

```powershell
.\run-moe-u3.cmd `
  -Model "B:\Ollama\Models\GGUF\Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.gguf" `
  -Plan "B:\Ollama\turboquant-plus-tqp_test\moe-u1-result\kat-static-plan.json"
```

Expected output ZIP:

```text
moe-u3-result-Kwaipilot_KAT-Coder-V2.5-Dev-Q5_K_M.zip
```

## U3 pass conditions

`u3-result.json` must report:

- `success=true`;
- `u2_accounting_ok=true`;
- `u2_packed_runtime_zero=true`;
- `baseline_exact_match=true`;
- `static_multiple_tokens=true`;
- `static_compact_loaded=true`;
- `static_packed_runtime_zero=true`;
- `cpu_branch_executed=true`;
- `accelerator_branch_executed=true`;
- every routed-weight copy check `true` because its measured counter is zero;
- `cache_type_k=turbo4`;
- `cache_type_v=turbo3`;
- `flash_attention=true`.

Return both generated ZIP files. No individual logs need to be selected manually.

## Avoiding unnecessary checks

A user checkpoint must not be requested merely because:

- a commit was created;
- CI is still running;
- a parser or report format changed;
- a synthetic/backend test can answer the question;
- the next implementation stage can safely continue without hardware evidence.

If a user checkpoint fails, diagnose and fix everything available in the returned ZIP before requesting another run.
