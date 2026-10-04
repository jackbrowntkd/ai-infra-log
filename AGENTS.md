# Repository Guidelines

A staged AI-infrastructure learning log. `README.md` is the index: environment facts, baselines, phase checklist. Every phase must leave a reproducible artifact.

## Project Structure & Module Organization

Each phase owns a top-level directory, currently only `phase0/`:

- `phase0/*.py` — measurement tools: `bench.py` (timing), `calibrate.py` (hardware peaks), `roofline.py` (plot).
- `phase0/cuda/` — CUDA sources (`*.cu`, `common.cuh`) and `build.ps1`; build output (`.exe/.lib/.obj`) is gitignored.
- `phase0/measured/*.json` — one file per kernel, consumed by `roofline.py`.
- `phase0/reports/`, `phase0/notes/` — committed evidence (`*_ncu.txt`, `*_qa.md`) and numbered notes (`NN_topic.md`).

New phases: `phaseN/` with the same layout plus a README checklist entry.

## Build, Test, and Development Commands

Run all commands from the repository root; `build.ps1` rejects other working directories.

```powershell
python phase0/calibrate.py      # -> phase0/hardware.json (~20 s)
python phase0/roofline.py       # -> phase0/reports/roofline.png
powershell -NoProfile -ExecutionPolicy Bypass -File phase0\cuda\build.ps1 phase0\cuda\vecadd.cu
.\phase0\cuda\vecadd.exe
ncu --set full -k "regex:vecadd_float4" --launch-skip 20 --launch-count 1 .\phase0\cuda\vecadd.exe  # admin shell
```

**优化一个 kernel 时，从这里开始**（把官方规则抽成按收益排序的清单）：

```powershell
python phase0/ncu_rules.py phase0/cuda/sgemm.exe -k sgemm_reg2 --skip 3 --out phase0/reports/sgemm_reg2_rules.md
```

`ncu` 的 INF / OPT / WRN 是 NVIDIA 规则引擎的判断，带量化收益（`Est. Speedup`），
应当作为优化的起点；本仓库自己的杠杆表（`notes/09`）只负责解释因果和排优先级。

`build.ps1` sources `vcvars64.bat` for `cl.exe` and forces `chcp 65001` so the Chinese path survives. Add `-LineInfo` when profiling, `-Extra @("-Xptxas","-v")` for register counts.

## Coding Style & Naming Conventions

- Python: 4-space indent, `from __future__ import annotations`, type hints, `pathlib` over `os.path`, `snake_case`. Open modules with a Chinese docstring ending in a `运行::` block showing the exact command.
- CUDA/C++: 4-space indent, `snake_case` kernels (`vecadd_float4`), `__restrict__` pointers, opening brace on the function line. Shared helpers like `CUDA_CHECK` live in `common.cuh`.
- Encoding is a hard rule: save `.cu`/`.cuh` as UTF-8 **with BOM** and keep every string literal in them pure ASCII — `nvcc` fails on non-ASCII literals. Chinese comments and docs are the norm.
- No formatter or linter is configured; match the surrounding code.

## Testing Guidelines

No test framework. Verification means reproduction: rerun the entry-point script and compare against the committed baseline.

- Timings must come from `bench.do_bench` — never `time.time()`, never a single run, always warm up and flush L2. `phase0/hardware.json` is the denominator for every utilization claim.
- Register a kernel via a `points[]` entry (`name`, `ms`, `gflops`, `gbps`, `ai`) in `phase0/measured/<name>.json`, then rerun `roofline.py`. Ship raw evidence beside each conclusion (`reports/<name>_ncu.txt` plus a short `*_qa.md`).
- This T600 has no Tensor Core, so fp16 is slower than fp32. Do not claim `mma`/FlashAttention speedups here.

### Evidence rules (each of these was earned by an actual mistake)

1. **Every metric named in a report must have a measured value in it.** If a table column, a checklist item, or a "you should look at `X`" sentence names a metric, that metric must appear in the same report's raw output. Naming `lts__throughput` while never collecting it is a bug, even when the conclusion happens to hold.
2. **One table, one capture.** All numbers in a single comparison table must come from the **same** profiler invocation. Collecting the same metric in two separate runs can differ by ~8 points (observed on `l1tex__throughput`). Re-run and re-capture rather than mixing.
3. **Timings come from standalone runs, utilization from the profiler.** `ncu` replays the kernel, which warms the cache; for cache-sensitive kernels a profiler Duration can be off by >20 % and even reverse the ranking between two versions (`sgemm_naive` vs `sgemm_tiled`). Use the profiler only to answer "which resource is busiest".
4. **State the level when quoting bytes.** Arithmetic intensity is meaningless until you say which level the bytes were counted at — DRAM and L2 give different numbers for the same kernel (see `notes/05` §10).

## Commit & Pull Request Guidelines

This directory is not a git repository yet, so there is no history to follow. If you initialize one:

- Commits: imperative subject, one artifact per commit, e.g. `phase0: add membound sweep and QA report`. Keep each claim and its raw output together.
- PRs: state the phase and question answered, the commands run, and the before/after numbers; link the report. Screenshots only for `reports/*.png`.
- Never commit `*.exe`, `*.lib`, `*.exp`, `*.obj`, or `__pycache__/` (already in `.gitignore`).
