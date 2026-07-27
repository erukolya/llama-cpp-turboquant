# User validation policy for static MoE placement

This document defines when work may continue without user involvement and when validation on the primary Windows/CUDA/KAT machine is required.

## Communication rule

The implementation agent must use one of these explicit statuses:

```text
USER NOT NEEDED
```

Work can continue using source inspection, unit tests, CI, synthetic models, static validation, or repository artifacts.

```text
USER CHECK REQUIRED
```

A result depends on the user's local KAT GGUF, RTX 5070 Ti, RAM/VRAM measurements, PCIe behavior, or real workload.

When a user check is required, provide:

1. one exact branch or release artifact;
2. one ready-to-run PowerShell command or script;
3. expected success markers;
4. exact output/log files to return;
5. no unrelated manual setup.

Do not ask the user to inspect source code, build partial commits manually, or repeat checks already covered by CI.

## Checks performed without the user

The agent should handle these independently:

- source and architecture analysis;
- planner implementation;
- JSON schema and parser tests;
- CLI parsing tests;
- CPU builds and cross-platform CI;
- server build checks;
- malformed-plan tests;
- synthetic tensor-layout validation;
- static memory accounting;
- documentation and progress tracking;
- code review and CI failure fixes.

## Hardware checkpoints

The target is to use no more than three user checkpoints for Version 1.

### U1 — KAT model-aware dry run

Trigger only after:

- all relevant CI is green;
- `--moe-expert-plan` is implemented;
- plan/model tensor manifest comparison is implemented;
- a Windows CUDA build artifact is available.

User runs one command that:

- loads the real KAT model;
- validates all routed tensor names, types, `ne[]`, `nb[]`, and expert strides;
- prints the intended CPU/VRAM split;
- changes no allocation and no inference behavior;
- writes a machine-readable validation report.

### U2 — Exclusive split-loader memory validation

Trigger only after:

- compact CPU and CUDA expert pools load successfully in CI/synthetic tests;
- original packed expert storage is released by design;
- model unload/reload tests pass where possible;
- a Windows CUDA artifact is available.

User runs one command that reports:

- total RAM;
- total VRAM;
- routed expert bytes in each tier;
- absence of a complete duplicated RAM expert bank;
- successful model startup and shutdown.

### U3 — Final correctness and performance validation

Trigger only after:

- mixed CPU/CUDA execution is implemented;
- correctness tests pass;
- CPU/GPU overlap is implemented;
- expert-weight transfer counters are available;
- the complete server path is ready.

The final user script should collect in one run where practical:

- baseline logits/output;
- static-placement logits/output;
- prompt-processing speed;
- token-generation speed;
- actual routing coverage;
- RAM and VRAM usage;
- routed-expert PCIe weight-transfer bytes;
- startup, generation, and shutdown logs.

## Avoiding unnecessary checks

A user checkpoint must not be requested merely because:

- a commit was created;
- CI is still running;
- a parser or report format changed;
- a synthetic test can answer the question;
- the next implementation stage can safely continue without hardware evidence.

If a user checkpoint fails, fix all issues that can be diagnosed from the returned logs before requesting another run.
