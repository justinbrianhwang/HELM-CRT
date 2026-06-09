# HELM: Heterogeneous Execution for Large Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**HELM** is a compiler and runtime for running large language models on consumer hardware — machines with a single GPU and CPU RAM. It microbenchmarks your hardware, compiles the model into static per-device FX subgraphs, and executes heterogeneously across CPU and GPU with a paged KV cache that offloads cold pages to CPU RAM during long-context decode.

---

## vs. Existing Tools

**vs. Accelerate** — Accelerate uses `device_map='auto'` which fills GPU from layer 0 upward with no runtime cost model. HELM microbenchmarks your actual hardware (FLOPS at decode/prefill regimes, memory bandwidth, PCIe throughput) and finds the partition that minimises decode latency under a roofline model. It compiles to isolated FX subgraphs rather than attaching per-layer forward hooks, eliminating Python-level synchronisation overhead at every boundary.

**vs. vLLM** — vLLM requires all model weights to fit in GPU VRAM and OOMs on models larger than VRAM capacity. HELM targets exactly this regime: models that cannot fit on a single GPU but need to run on one anyway.

**vs. FlexGen** — FlexGen overlaps weight streaming with compute, which only helps at batch sizes in the hundreds. HELM keeps all weights resident in CPU RAM + GPU VRAM statically; the only cross-device traffic is a single activation tensor at the stage boundary, once per decode step.

**vs. DeepSpeed ZeRO-Inference** — DeepSpeed streams weights from CPU per layer during inference, incurring O(L) PCIe round-trips per generated token. HELM places weights statically and only transfers activations at the stage boundary — no per-layer weight movement during decode.

---

## How It Works

```
Model (nn.Module)
      │
      ▼
 [1] FX Tracing          — Capture a single decode graph (seq_len=1); the same
      │                     compiled graph is reused for both prefill and decode
      ▼
 [2] HelmGraph IR         — Lift FX nodes into typed IR with shape, FLOP, and
      │                     byte-cost metadata (shape propagation + fallback estimator)
      ▼
 [3] HybridAnalyzer       — Annotate each node with param bytes, activation bytes,
      │                     and KV bytes per token; aggregate per transformer block
      ▼
 [4] PartitionUnitBuilder — Group nodes into coarse units: one embedding unit,
      │                     L transformer-block units, one output-projection unit
      ▼
 [5] DeviceProfiler       — Microbenchmark actual GPU/CPU FLOPS (decode GEMV regime
      │                     + prefill GEMM regime), DRAM bandwidth, and PCIe H2D/D2H;
      │                     results cached across compile() calls
      ▼
 [6] StrategySelector     — Score all O(L) feasible CPU→GPU split points under the
      │                     roofline cost model (separate projection + attention
      │                     rooflines; L3 cache hierarchy for short-context KV;
      │                     GPU flash-attention activation traffic = 0);
      │                     return the split minimising the configured objective
      ▼
 [7] StageFXBuilder       — Fragment the FX graph into per-device standalone
      │                     GraphModules at the chosen split boundary
      ▼
 [8] PipelineRuntime      — Prefill pass → autoregressive decode loop (batch ≥ 1);
      │                     both phases reuse the same compiled stage subgraphs
      ▼
 [9] StageRuntimeExecutor — Execute stages in sequence; transfer activation tensor
      │                     across device boundary; pre-position weights one submodule
      │                     at a time to avoid peak-RAM spikes during load
      ▼
[10] KVOffloadManager     — Paged KV cache with GPU watermark; evict oldest pages
                            to CPU pinned RAM; stream back page-by-page during decode
                            via async H2D + online-softmax streaming attention
```

**StrategySelector** evaluates three plan classes: (1) all-GPU, (2) all feasible CPU→GPU 2-stage splits, (3) all-CPU. The exhaustive search runs in O(L) time and takes <1 ms.

**KVOffloadManager** patches attention `forward` at the class level (no FX graph changes needed). Supports `batch_size > 1`: one `KVCacheManager` per batch item sharing a single `KVAllocator` pool. Eviction policy: oldest page (by `start_token`) first, protecting the active tail page.

---

## Results (RTX 3090, 24 GB VRAM, batch=1, fp16)

Decode throughput at output length 128:

| Model | HELM | Accelerate | DeepSpeed | vLLM |
|---|---|---|---|---|
| Qwen3-4B  | 20.9 tok/s | 26.8 tok/s | 24.3 tok/s | --- (not tested) |
| Qwen3-8B  | 19.8 tok/s | 25.1 tok/s | 23.8 tok/s | **50.6 tok/s** |
| Qwen3-14B | **7.3 tok/s** | 1.3 tok/s | OOM | OOM |
| Qwen3-32B | **1.1 tok/s** | --- | OOM | OOM |

Maximum decode length before OOM:

| Model | HELM | Accelerate | DeepSpeed | vLLM |
|---|---|---|---|---|
| Qwen3-4B  | **32,768** | 4,096 | 4,096 | --- |
| Qwen3-8B  | **16,384** | 4,096 | 4,096 | 8,192 |
| Qwen3-14B | **4,096**  | 128   | OOM   | OOM  |
| Qwen3-32B | **1,024**  | ---   | OOM   | OOM  |

Key observations:
- On **small models that fit in VRAM** (4B/8B on 3090), vLLM and Accelerate are faster — HELM's CPU stage adds overhead that pays off only when it enables feasibility.
- On **models that exceed VRAM** (14B/32B on 3090; all models on an 8 GB GPU), HELM is the only backend that runs. It achieves **5.6× higher throughput** than Accelerate on 14B and enables 32B inference where every other backend OOMs.
- HELM's **paged KV offload** extends context to 32K tokens on 4B and 16K on 8B — 8× and 4× longer than Accelerate/DeepSpeed.

---

## Installation

**Requirements:** Linux **or Windows via WSL2**, Python 3.10+, NVIDIA driver ≥ 525 + CUDA ≥ 11.8, `uv`

> Reproduced on this fork: **WSL2 (Ubuntu 22.04)** on Windows 11 · NVIDIA **RTX 5090** (32 GB, Blackwell `sm_120`) · CUDA 13.2 driver · 62 GB RAM · GCC 11.4.

```bash
git clone https://github.com/justinbrianhwang/HELM-CRT.git
cd HELM-CRT
uv sync
```

The AVX2+F16C CPU kernel is built automatically via a C++ extension on first use. Requires GCC ≥ 9 and a CPU with AVX2 support (Intel Haswell+ / AMD Ryzen+).

Two sets of extras are needed in practice and install into the same environment:

```bash
# ninja: torch JIT-builds the AVX2 CPU kernel and needs it
# sentencepiece + protobuf: required by the Llama / Mistral tokenizers
uv pip install ninja sentencepiece protobuf
```

On a Blackwell GPU (RTX 50-series) the `cu128` PyTorch wheel pinned in `pyproject.toml` already ships `sm_120` support — no extra steps. The `helm` CLI entry point runs the single-model driver in `experiments/dev_pipeline.py`.

---

## Usage

### Quick start — auto partition

```bash
uv run helm \
    --model Qwen/Qwen3-4B \
    --mode execute_stagewise \
    --compiler-plan auto \
    --max-new-tokens 64 \
    --kv-offload
```

### Manual partition

Specify exactly which layers run on CPU and which on GPU. Ranges are **inclusive**.

```bash
uv run helm \
    --model Qwen/Qwen3-14B \
    --mode execute_stagewise \
    --compiler-plan manual \
    --compiler-cpu-layers 0:7 \
    --compiler-gpu-layers 8:47 \
    --max-new-tokens 128 \
    --kv-offload
```

### Inspect the partition plan without executing

```bash
uv run helm \
    --model Qwen/Qwen3-8B \
    --mode plan \
    --compiler-plan auto \
    --print-plan
```

---

## Paper Experiments

The full experiment suite runs with a single command:

```bash
bash experiments/run_paper_experiments.sh
```

Self-contained — no arguments required. It detects hardware, selects feasible backends, and runs every combination of **4 models × 4 backends × 6 experiment sections**:

| Section | What it measures |
|---|---|
| **E — Feasibility** | Does the model load? Peak GPU/CPU memory at load time |
| **F — Max decode length** | Longest output before OOM |
| **A — Latency sweep** | TTFT, decode tok/s — p50/p95/p99 across 20 requests |
| **B — Throughput** | Aggregate tok/s at batch sizes 1/2/4/8 |
| **C — Ablations** | batch size, context length, AVX on/off, KV offload on/off, CPU threads |
| **D — LM quality** | MMLU, HellaSwag, ARC-Easy via lm-evaluation-harness |

Models: `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`

**Crash recovery:** each section writes a `.done` sentinel on completion. Re-running skips completed sections.

**Environment overrides:**

```bash
QUICK=1              # 5 requests, shorter sequences — smoke-test on any machine
NO_LM_EVAL=1         # skip Section D
SECTIONS=E,F,A       # run only specific sections
OUT_ROOT=/scratch/results
```

Results land in `experiments/results/<timestamp>/` with a `SUMMARY.md` containing paper-ready tables.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model name or local path |
| `--mode` | `plan` | `dry_run` · `plan` · `lower` · `execute_stagewise` · `baseline` |
| `--compiler-plan` | `auto` | `auto` (profiled) or `manual` |
| `--compiler-cpu-layers` | — | Layer range for CPU, e.g. `0:7` |
| `--compiler-gpu-layers` | — | Layer range for GPU, e.g. `8:47` |
| `--kv-offload` | off | Enable paged KV cache with CPU offloading |
| `--max-new-tokens` | `8` | Tokens to generate |
| `--dtype` | `float16` | `float16` · `bfloat16` · `float32` |
| `--batch-size` | `1` | Requests processed in parallel |
| `--cpu-threads` | auto | CPU thread count for PyTorch ops |
| `--print-plan` | off | Print full partition plan JSON |

---

## License

[Apache License 2.0](LICENSE)

---

# Extended Experiments — Additional Model Coverage (this fork)

> Everything **above** this line is the original **HELM** work by **MPS LAB**.
> Everything **below** documents **additional experiments I ran on top of it** — extending HELM to more HuggingFace model families and fixing what was needed to make them run.
> — **justinbrianhwang**, 2026

## Setup

Run on **WSL2 (Ubuntu 22.04)** / Windows 11, single **NVIDIA RTX 5090** (32 GB, Blackwell `sm_120`), 128 GB system RAM, CUDA 13.2 driver, PyTorch 2.10 (`cu128`), `transformers` 4.57.

## What I did

- **Verified HELM end-to-end on 5 models from 4 different families** (Mistral, Llama, DeepSeek-distill, Qwen) — each compiles, partitions, and generates coherent text under `--kv-offload`.
- **Fixed Mistral support** (`helm/runtime/kv_offload.py`). On the `--kv-offload` decode path, Mistral was routed to the Llama attention patch, which targets `LlamaAttention`; a Mistral model uses its own `MistralAttention` class, so the patch never fired. Added a dedicated `_make_mistral_forward` + dispatch branch. Also fixed `KVOffloadConfig.from_model`: Mistral-7B-v0.3's config *defines* `head_dim` but leaves it `None`, which crashed cache sizing — now falls back via `getattr(cfg, k, None) or <computed>`.
- **Fixed the broken `helm` CLI entry point** (`helm/cli.py`) — it pointed at a `benchmarks/run_benchmark.py` that isn't in the tree; repointed it to the actual driver `experiments/dev_pipeline.py`.
- **Added decode-timing instrumentation** (`helm/runtime/pipeline_runtime.py`, `experiments/dev_pipeline.py`) to report TTFT and decode tok/s.

## Results (RTX 5090, 32 GB, batch=1, fp16, `--kv-offload`)

Single-run, indicative numbers via `helm --mode execute_stagewise --compiler-plan auto`:

| Model | Family | HELM partition (auto) | TTFT | Decode tok/s |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | Mistral | all-GPU (14.5 GB) | 871 ms | 39.0 |
| Llama-3.1-8B-Instruct | Llama | all-GPU (16.1 GB) | 126 ms | 32.8 |
| DeepSeek-R1-Distill-Llama-8B | Llama | all-GPU (16.1 GB) | 108 ms | 42.7 |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2 | all-GPU (16.3 GB) | 115 ms | 44.7 |
| **Qwen3-32B** | Qwen3 | **CPU+GPU split** (33.7 GB CPU + 31.8 GB GPU) | 9.1 s | 1.31 |

Key points:
- The four 7–8B models fit in 32 GB VRAM, so HELM's cost model correctly keeps them **all-GPU**.
- **Qwen3-32B is the headline:** the model is **65.5 GB in fp16** — it does **not** fit in 32 GB VRAM. HELM ruled all-GPU infeasible and split it across **CPU (33.7 GB) + GPU (31.8 GB)**, running the CPU stage through the AVX2+F16C GEMV kernel (`Patched 231 nn.Linear(s)`). Every other backend (vLLM / Accelerate / DeepSpeed) OOMs in this regime — HELM is the only one that runs it on a single 32 GB GPU.

## Reproduce

```bash
uv run helm --model Qwen/Qwen3-32B \
    --mode execute_stagewise --compiler-plan auto \
    --max-new-tokens 16 --kv-offload
```

Gated models (Llama, Mistral) need a Hugging Face token with the licenses accepted; DeepSeek-distill and Qwen are open.
