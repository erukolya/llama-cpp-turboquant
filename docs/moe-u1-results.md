# KAT U1 validation results

Date: 2026-07-27

Both uploaded U1 packages passed schema-v2 strict validation against the real runtime tensors on RTX 5070 Ti.

## Common validation result

- architecture: Qwen3.5 MoE / KAT-Coder-V2.5-Dev
- routed layers: 40
- routed experts per layer: 256
- logical routed experts: 10,240
- routed tensors in manifest: 120
- placement budget: 10,338 MiB
- exact tensor names, types, `ne[4]`, `nb[4]`, sizes, expert ranges and `nb[2]` strides matched
- dry-run allocation and inference remained unchanged

## Q5_K_M

| Metric | Value |
|---|---:|
| Routed expert bank | 21.289 GiB |
| Experts selected for VRAM | 4,863 / 10,240 |
| Selected routed bytes | 10,337.664 MiB |
| Cold routed bytes remaining in RAM | 11.194 GiB |
| Estimated GPU routing coverage | 71.242% |
| Estimated CPU routing share | 28.758% |
| Model fingerprint | `sampled-fnv1a64:66d21554683500ba` |

## Q4_K_M

| Metric | Value |
|---|---:|
| Routed expert bank | 18.164 GiB |
| Experts selected for VRAM | 5,725 / 10,240 |
| Selected routed bytes | 10,337.695 MiB |
| Cold routed bytes remaining in RAM | 8.069 GiB |
| Estimated GPU routing coverage | 78.047% |
| Estimated CPU routing share | 21.953% |
| Model fingerprint | `sampled-fnv1a64:d1859e626beb3c0e` |

## Comparison

- The Q4 hot set is a strict superset of the Q5 hot set.
- All 4,863 Q5-selected `(layer, expert)` pairs are also selected for Q4.
- Q4 adds 862 more experts under the same routed-expert VRAM budget.
- Q4 raises estimated GPU routing coverage by 6.806 percentage points.
- Q4 reduces estimated CPU-routed selections by 23.665% relative to Q5.
- Q4 reduces the routed expert bank size by 14.679%.
- Under exclusive placement, Q4 leaves about 3.125 GiB less routed-expert data in RAM.

These are placement and routing estimates, not measured tokens/s or quality results. Q4 is expected to be more favorable for the static split runtime, but final throughput and quality must be measured after V1.3/V1.4 are implemented.

## Gate decision

U1 is accepted for both Q5_K_M and Q4_K_M. V1.2 exclusive split storage may proceed. The next user hardware checkpoint is U2 after compact CPU/CUDA pools are implemented and an artifact is ready.
