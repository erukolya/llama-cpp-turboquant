# MoE U2 model-load validation results

Date: 2026-07-28

Hardware:

- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB
- CUDA user-mode driver: 13.3
- system RAM: 34,281,717,760 bytes
- model architecture: `qwen35moe`
- model layers: 40
- routed experts: 256 per layer, 10,240 total

U2 validates model-only exclusive compact storage. It does not create an inference context or execute tokens.

## Q5_K_M

**Result: accepted.**

| Metric | Value |
|---|---:|
| CPU experts | 5,377 |
| GPU experts | 4,863 |
| CPU compact tensors | 120 |
| GPU compact tensors | 120 |
| Planned CPU routed bytes | 12,019,130,368 |
| Actual CPU routed bytes | 12,019,130,368 |
| Planned GPU routed bytes | 10,839,826,432 |
| Actual GPU routed bytes | 10,839,826,432 |
| Planned total routed bytes | 22,858,956,800 |
| Actual total routed bytes | 22,858,956,800 |
| Packed routed runtime tensors | 0 |

Observed success markers:

```text
moe_u2: loaded
moe_u2: packed_runtime_tensors=0
moe_u2: accounting=ok
moe_u2: unloaded
```

This proves for Q5_K_M that the compact CPU/CUDA pools load exact planned bytes, the original packed routed tensors are not allocated, and unload completes.

## Q4_K_M

**Result: loader behavior passed, package accounting result invalid because the Q5 plan was supplied.**

The result archive records this plan path:

```text
B:\Ollama\turboquant-plus-tqp_test\moe-u1-result\kat-static-plan.json
```

Its expected byte totals are identical to the accepted Q5 plan, while the loaded Q4 model produced the smaller Q4 compact pools:

| Metric | Plan supplied to checker | Actual Q4 load |
|---|---:|---:|
| CPU routed bytes | 12,019,130,368 | 10,272,202,752 |
| GPU routed bytes | 10,839,826,432 | 9,231,310,848 |
| Total routed bytes | 22,858,956,800 | 19,503,513,600 |
| Packed routed runtime tensors | — | 0 |

The checker therefore correctly rejected the run with:

```text
moe_u2: error=compact CPU logical bytes differ from the plan
```

The Q4 load still confirms that 120 CPU and 120 GPU compact tensors were created and that `packed_runtime_tensors=0`. It is not accepted as the exact U2 accounting gate until rerun with the matching Q4 plan.

Newer U2 validator code checks the sampled model fingerprint before backend initialization and model allocation, so a Q4/Q5 plan mismatch is rejected before allocating compact pools.

## Decision

- Keep both Q5_K_M and Q4_K_M as supported target quants.
- Q5_K_M has passed the complete U2 gate.
- Q4_K_M remains pending only for exact matching-plan accounting; its exclusive compact loader path already behaved as intended.
- Continue V1.3 mixed CPU/CUDA execution work without waiting for another hardware run.
