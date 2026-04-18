"""
experiments/paper_bench.py
==========================
Comprehensive benchmark for HELM paper experiments.

Covers:
  Backends   : vLLM, Accelerate (HF), DeepSpeed-Inference (CPU offload), HELM
  LLM metrics: TTFT, per-token decode latency, E2E latency, tok/s
  Statistics : p50 / p95 / p99 across --num-requests independent requests
  Compiler   : compilation time, stage plan, cost-model prediction vs actual
  Runtime    : peak GPU MB, peak CPU MB
  Throughput : concurrent requests/sec and aggregate tok/s
  Feasibility: which backends can load the model (fits_in_memory), peak memory at load
  Max decode : longest decode sequence each backend sustains before OOM/timeout
  Ablations  : configurable via --ablation flag

Usage examples
--------------
# Full paper experiment on one machine
uv run python experiments/paper_bench.py --model Qwen/Qwen2.5-7B-Instruct

# Specific backends only
uv run python experiments/paper_bench.py --backends helm accelerate

# Ablation: batch sizes
uv run python experiments/paper_bench.py --ablation batch_size --backends helm

# Skip quality benchmarks (faster)
uv run python experiments/paper_bench.py --no-lm-eval
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ─── Memory helpers ──────────────────────────────────────────────────────────

def _free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0

def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _cpu_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return 0.0

def _ram_used_pct() -> float:
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


# ─── Timeout helper ──────────────────────────────────────────────────────────

class _BenchTimeout(Exception):
    pass

def _set_alarm(seconds: int):
    if hasattr(signal, "SIGALRM"):
        def _handler(sig, frame):
            raise _BenchTimeout(f"run timed out after {seconds}s")
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)

def _cancel_alarm():
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    """Per-request measurement."""
    ttft_s:          float = 0.0    # time-to-first-token (= prefill time)
    decode_s:        float = 0.0    # total decode phase time
    e2e_s:           float = 0.0    # ttft + decode
    input_tokens:    int   = 0
    output_tokens:   int   = 0
    tok_per_s:       float = 0.0    # output_tokens / e2e_s
    decode_tok_per_s: float = 0.0   # output_tokens / decode_s
    peak_gpu_mb:     float = 0.0
    peak_cpu_mb:     float = 0.0
    status:          str   = "success"
    error_msg:       str   = ""


@dataclass
class BenchStats:
    """Aggregated statistics over N requests."""
    backend:      str   = ""
    ablation_tag: str   = ""          # e.g. "batch=4", "ctx=512"
    n_requests:   int   = 0
    n_success:    int   = 0

    # Latency statistics (seconds)
    ttft_p50:     float = 0.0
    ttft_p95:     float = 0.0
    ttft_p99:     float = 0.0
    ttft_mean:    float = 0.0

    decode_lat_p50: float = 0.0     # per-token decode latency (ms)
    decode_lat_p95: float = 0.0
    decode_lat_p99: float = 0.0
    decode_lat_mean: float = 0.0

    e2e_p50:      float = 0.0
    e2e_p95:      float = 0.0
    e2e_p99:      float = 0.0

    # Throughput
    tok_per_s_mean:       float = 0.0
    decode_tok_per_s_mean: float = 0.0
    throughput_req_per_s: float = 0.0  # filled by throughput_sweep()
    throughput_tok_per_s: float = 0.0

    # Memory
    peak_gpu_mb_mean:  float = 0.0
    peak_cpu_mb_mean:  float = 0.0

    # HELM compiler metrics (only set for helm backend)
    compile_time_s:        float = 0.0
    stage_plan:            str   = ""   # e.g. "stage0@cpu(14u), stage1@cuda(14u)"
    cost_model_decode_ms:  float = 0.0  # predicted per-token decode latency
    cost_model_prefill_ms: float = 0.0  # predicted prefill latency

    raw_requests: List[dict] = field(default_factory=list)


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100) * (len(data_sorted) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(data_sorted) - 1)
    return data_sorted[lo] + (idx - lo) * (data_sorted[hi] - data_sorted[lo])


def _aggregate(results: List[RequestResult], backend: str, ablation_tag: str = "") -> BenchStats:
    ok = [r for r in results if r.status == "success"]
    stats = BenchStats(backend=backend, ablation_tag=ablation_tag,
                       n_requests=len(results), n_success=len(ok))
    if not ok:
        return stats

    ttft   = [r.ttft_s for r in ok]
    e2e    = [r.e2e_s for r in ok]
    decs   = [r.decode_s for r in ok]
    dtoks  = [r.decode_tok_per_s for r in ok]
    toks   = [r.tok_per_s for r in ok]
    gpus   = [r.peak_gpu_mb for r in ok]
    cpus   = [r.peak_cpu_mb for r in ok]

    # TTFT
    stats.ttft_p50  = _percentile(ttft, 50) * 1000   # ms
    stats.ttft_p95  = _percentile(ttft, 95) * 1000
    stats.ttft_p99  = _percentile(ttft, 99) * 1000
    stats.ttft_mean = statistics.mean(ttft) * 1000

    # Per-token decode latency (ms per token)
    out_tokens = [r.output_tokens for r in ok]
    per_tok = [d / t * 1000 for d, t in zip(decs, out_tokens) if t > 0]
    stats.decode_lat_p50  = _percentile(per_tok, 50)
    stats.decode_lat_p95  = _percentile(per_tok, 95)
    stats.decode_lat_p99  = _percentile(per_tok, 99)
    stats.decode_lat_mean = statistics.mean(per_tok) if per_tok else 0.0

    # E2E
    stats.e2e_p50 = _percentile(e2e, 50) * 1000
    stats.e2e_p95 = _percentile(e2e, 95) * 1000
    stats.e2e_p99 = _percentile(e2e, 99) * 1000

    # Throughput
    stats.tok_per_s_mean        = statistics.mean(toks)
    stats.decode_tok_per_s_mean = statistics.mean(dtoks)

    # Memory
    stats.peak_gpu_mb_mean = statistics.mean(gpus)
    stats.peak_cpu_mb_mean = statistics.mean(cpus)

    stats.raw_requests = [asdict(r) for r in ok]
    return stats


# ─── Base backend interface ───────────────────────────────────────────────────

class Backend:
    name: str = "base"

    def setup(self) -> bool:
        raise NotImplementedError

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        raise NotImplementedError

    def teardown(self):
        pass

    def run_n(
        self,
        prompts: List[str],
        output_len: int,
        input_len: int,
        timeout_s: int = 300,
    ) -> List[RequestResult]:
        results = []
        for i, p in enumerate(prompts):
            _set_alarm(timeout_s)
            try:
                r = self.run_one(p, output_len, input_len)
            except _BenchTimeout as e:
                r = RequestResult(status="timeout", error_msg=str(e))
            except Exception as e:
                r = RequestResult(status="error", error_msg=str(e)[:300])
            finally:
                _cancel_alarm()
            results.append(r)
        return results


# ─── vLLM backend ────────────────────────────────────────────────────────────

class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model_name: str, dtype_str: str,
                 gpu_memory_utilization: float = 0.90):
        self.model_name = model_name
        self.dtype_str = dtype_str
        self.gpu_util = gpu_memory_utilization
        self._llm = None
        self._sampling_params = None

    def setup(self) -> bool:
        try:
            from vllm import LLM, SamplingParams
            print("[vLLM] Initialising engine …")
            self._llm = LLM(
                model=self.model_name,
                dtype=self.dtype_str,
                gpu_memory_utilization=self.gpu_util,
                max_model_len=4096,
            )
            self._sampling_params = SamplingParams
            print("[vLLM] Ready.")
            return True
        except Exception as e:
            print(f"[vLLM] Setup failed: {e}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=output_len, temperature=0.0)

        _reset_peak()
        t0 = time.perf_counter()
        out = self._llm.generate([prompt], sp)
        _sync()
        t1 = time.perf_counter()

        res = out[0]
        in_toks  = len(res.prompt_token_ids)
        out_toks = len(res.outputs[0].token_ids)
        e2e = t1 - t0

        # vLLM 0.17.1+: first_token_latency is pre-computed (arrival→first token)
        ttft = 0.0
        if hasattr(res, "metrics") and res.metrics is not None:
            ttft = getattr(res.metrics, "first_token_latency", 0.0) or 0.0

        return RequestResult(
            ttft_s=ttft,
            decode_s=max(e2e - ttft, 0.0),
            e2e_s=e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / e2e if e2e > 0 else 0.0,
            decode_tok_per_s=out_toks / max(e2e - ttft, 1e-6),
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        del self._llm
        self._llm = None
        _free_memory()


# ─── Accelerate backend ───────────────────────────────────────────────────────

class AccelerateBackend(Backend):
    name = "accelerate"

    def __init__(self, model_name: str, dtype_str: str):
        self.model_name = model_name
        self.dtype_str  = dtype_str
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_str, torch.float16)
        self._model = None
        self._tokenizer = None

    def setup(self) -> bool:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("[Accelerate] Loading model …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            print("[Accelerate] Ready.")
            return True
        except Exception as e:
            print(f"[Accelerate] Setup failed: {e}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        tok = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=input_len,
        )
        inp = tok["input_ids"]
        device = next(self._model.parameters()).device
        inp = inp.to(device)

        _reset_peak()

        # Time prefill (TTFT) separately via one forward pass
        with torch.no_grad():
            t_pre0 = time.perf_counter()
            out_ids = self._model.generate(
                inp,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )
            _sync()
            t_pre1 = time.perf_counter()
            ttft = t_pre1 - t_pre0

            # Full decode
            t_dec0 = time.perf_counter()
            full_ids = self._model.generate(
                inp,
                max_new_tokens=output_len,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )
            _sync()
            t_dec1 = time.perf_counter()

        in_toks  = inp.shape[1]
        out_toks = full_ids.shape[1] - in_toks
        e2e = t_dec1 - t_dec0

        return RequestResult(
            ttft_s=ttft,
            decode_s=e2e,
            e2e_s=ttft + e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + e2e) if (ttft + e2e) > 0 else 0.0,
            decode_tok_per_s=out_toks / e2e if e2e > 0 else 0.0,
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        del self._model
        self._model = None
        _free_memory()


# ─── DeepSpeed backend ────────────────────────────────────────────────────────

class DeepSpeedBackend(Backend):
    """DeepSpeed-Inference with CPU offloading (ZeRO-Inference)."""
    name = "deepspeed"

    def __init__(self, model_name: str, dtype_str: str, cpu_offload: bool = True):
        self.model_name  = model_name
        self.dtype_str   = dtype_str
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_str, torch.float16)
        self.cpu_offload = cpu_offload
        self._model      = None
        self._tokenizer  = None

    def setup(self) -> bool:
        try:
            import deepspeed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("[DeepSpeed] Loading model …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load to CPU first for ZeRO-Inference offload
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            base.eval()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DeepSpeed] Initialising inference engine (cpu_offload={self.cpu_offload}) …")

            if self.cpu_offload and torch.cuda.is_available():
                # ZeRO-Inference: weights stay on CPU, only active layers on GPU
                self._model = deepspeed.init_inference(
                    base,
                    dtype=self.dtype,
                    enable_cuda_graph=False,
                    replace_with_kernel_inject=False,  # safer for Qwen
                    injection_policy=None,
                    mp_size=1,
                )
            else:
                self._model = deepspeed.init_inference(
                    base,
                    dtype=self.dtype,
                    enable_cuda_graph=False,
                )
            print("[DeepSpeed] Ready.")
            return True
        except ImportError:
            print("[DeepSpeed] deepspeed not installed — skipping")
            return False
        except Exception as e:
            print(f"[DeepSpeed] Setup failed: {e}\n{traceback.format_exc()}")
            return False

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        tok = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=input_len,
        )
        inp = tok["input_ids"]
        if torch.cuda.is_available():
            inp = inp.cuda()

        _reset_peak()

        with torch.no_grad():
            # TTFT: generate 1 token
            t_pre0 = time.perf_counter()
            _ = self._model.generate(
                inp, max_new_tokens=1, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
            _sync()
            t_pre1 = time.perf_counter()
            ttft = t_pre1 - t_pre0

            # Full generation
            t0 = time.perf_counter()
            out_ids = self._model.generate(
                inp, max_new_tokens=output_len, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
            _sync()
            t1 = time.perf_counter()

        in_toks  = inp.shape[1]
        out_toks = out_ids.shape[1] - in_toks
        e2e = t1 - t0

        return RequestResult(
            ttft_s=ttft,
            decode_s=e2e,
            e2e_s=ttft + e2e,
            input_tokens=in_toks,
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + e2e) if (ttft + e2e) > 0 else 0.0,
            decode_tok_per_s=out_toks / e2e if e2e > 0 else 0.0,
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        del self._model
        self._model = None
        _free_memory()


# ─── HELM backend ─────────────────────────────────────────────────────────────

class HelmBackend(Backend):
    """HELM heterogeneous inference — auto partition mode."""
    name = "helm"

    def __init__(
        self,
        model_name:    str,
        dtype_str:     str,
        kv_offload:    bool = True,
        cpu_threads:   int  = 8,
        input_len:     int  = 64,
        batch_size:    int  = 1,
        disable_avx:   bool = False,   # ablation: disable AVX kernel
        disable_async: bool = False,   # ablation: disable async KV prefetch
    ):
        self.model_name   = model_name
        self.dtype_str    = dtype_str
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype_str, torch.float16)
        self.kv_offload   = kv_offload
        self.cpu_threads  = cpu_threads
        self.input_len    = input_len
        self.batch_size   = batch_size
        self.disable_avx  = disable_avx
        self.disable_async = disable_async

        self._model           = None
        self._tokenizer       = None
        self._runtime         = None
        self._kv_offload_mgr  = None
        self._run_lock        = threading.Lock()  # HELM is single-request; serialize concurrent calls
        self._decode_wrapper  = None

        # Compiler metrics captured during setup
        self.compile_time_s        = 0.0
        self.stage_plan            = ""
        self.cost_model_decode_ms  = 0.0
        self.cost_model_prefill_ms = 0.0

    def setup(self) -> bool:
        sys.path.insert(0, os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..')))
        try:
            from experiments.dev_pipeline import (
                load_model_and_tokenizer,
                capture_fx_graph,
                capture_decode_fx_graph,
                configure_cpu_threads,
            )
            from helm.compiler.compiler import HelmCompileOptions, compile_graph
            from helm.compiler.importers.decode_tracer import DecodeTracer
            from helm.runtime.executor import StageRuntimeExecutor
            from helm.runtime.pipeline_runtime import PipelineRuntime
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import psutil

            configure_cpu_threads(self.cpu_threads)

            if self.disable_avx:
                os.environ["HELM_DISABLE_AVX"] = "1"
                print("[HELM] AVX kernel disabled (ablation)")
            if self.disable_async:
                os.environ["HELM_DISABLE_ASYNC_KV"] = "1"
                print("[HELM] Async KV prefetch disabled (ablation)")

            print("[HELM] Loading model …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            gpu_budget = None
            if torch.cuda.is_available():
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                gpu_budget = f"{int(gpu_total_mb * 0.80)}MiB"
            ram_total_mb = psutil.virtual_memory().total / 1024 ** 2
            cpu_budget = f"{int(ram_total_mb - 2048)}MiB"

            max_memory: Dict = {"cpu": cpu_budget}
            if gpu_budget and torch.cuda.is_available():
                max_memory[0] = gpu_budget

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
                use_cache=False,
            )
            self._model.eval()

            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(self._model)
            except ImportError:
                pass

            prompt_tok = self._tokenizer(
                "Benchmark prompt for compilation.",
                return_tensors="pt",
                truncation=True,
                max_length=self.input_len,
            )
            input_ids = prompt_tok["input_ids"]
            seq_len = input_ids.shape[1]

            position_ids  = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            cache_position = torch.arange(seq_len, dtype=torch.long)
            min_val = torch.finfo(torch.float16).min
            causal  = torch.triu(
                torch.full((seq_len, seq_len), min_val, dtype=torch.float16), diagonal=1)
            inv_mask = (1.0 - prompt_tok["attention_mask"].float()) * min_val
            attn_mask = causal[None, None, :, :] + inv_mask[:, None, None, :]
            dummy_inputs = (input_ids, attn_mask, position_ids, cache_position)

            _workload = {
                "batch_size": self.batch_size,
                "prefill_seq_len": seq_len,
                "decode_context_len": seq_len,
                "decode_tokens": 128,
                "dtype_size": 2,
            }

            print("[HELM] Tracing and compiling (auto partition mode) …")
            t_compile0 = time.perf_counter()

            prefill_gm = capture_fx_graph(
                self._model, dummy_inputs, run_dir=None, allow_fallback=True)
            prefill_opts = HelmCompileOptions(
                mode="both", objective="decode_latency",
                plan_mode="auto",
                lower_stages=True, graph_kind="prefill",
                model_name=self.model_name, workload=_workload,
                kv_offload=self.kv_offload,
            )
            prefill_artifact = compile_graph(
                gm=prefill_gm, example_inputs=dummy_inputs,
                model=self._model, tokenizer=self._tokenizer,
                options=prefill_opts, artifacts_dir=None,
            )

            dec_dummy = DecodeTracer.build_dummy_inputs(
                device="cpu", batch_size=1, dtype=torch.float16)
            decode_gm = capture_decode_fx_graph(self._model, dec_dummy, run_dir=None)
            self._decode_wrapper = getattr(self._model, "_helm_decode_wrapper", None)

            decode_opts = HelmCompileOptions(
                mode="both", objective="decode_latency",
                plan_mode="auto",
                lower_stages=True, graph_kind="decode",
                model_name=self.model_name, workload=_workload,
            )
            decode_artifact = compile_graph(
                gm=decode_gm, example_inputs=dec_dummy,
                model=self._model, tokenizer=self._tokenizer,
                options=decode_opts,
                partition_plan_override=prefill_artifact.partition_plan,
                artifacts_dir=None,
            )

            t_compile1 = time.perf_counter()
            self.compile_time_s = t_compile1 - t_compile0

            # Capture stage plan and cost model predictions
            plan = prefill_artifact.partition_plan
            if plan is not None:
                self.stage_plan = ", ".join(
                    f"stage{s.stage_id}@{s.device_id}({len(s.units)}u)"
                    for s in plan.stages
                )
            if hasattr(prefill_artifact, "plan_cost") and prefill_artifact.plan_cost is not None:
                cost = prefill_artifact.plan_cost
                self.cost_model_decode_ms  = getattr(cost, "decode_token_latency_s", 0.0) * 1000
                self.cost_model_prefill_ms = getattr(cost, "prefill_latency_s", 0.0) * 1000

            print(f"[HELM] Compilation done in {self.compile_time_s:.1f}s")
            print(f"[HELM] Stage plan: {self.stage_plan}")
            print(f"[HELM] Cost model: decode={self.cost_model_decode_ms:.1f}ms "
                  f"prefill={self.cost_model_prefill_ms:.0f}ms")

            if self.kv_offload:
                from helm.runtime.kv_offload import KVOffloadManager, KVOffloadConfig
                kv_cfg = KVOffloadConfig.from_model(self._model)
                self._kv_offload_mgr = KVOffloadManager(
                    self._model, kv_cfg, batch_size=self.batch_size)

            prefill_exec = StageRuntimeExecutor(prefill_artifact.stage_graphs)
            decode_exec  = StageRuntimeExecutor(decode_artifact.stage_graphs)
            self._runtime = PipelineRuntime(
                prefill_exec, decode_exec,
                tokenizer=self._tokenizer,
                dtype=self.dtype,
                kv_offload_mgr=self._kv_offload_mgr,
                decode_wrapper=self._decode_wrapper,
            )
            print("[HELM] Runtime ready.")
            return True

        except Exception as e:
            print(f"[HELM] Setup failed:\n{traceback.format_exc()}")
            return False

    def _reset_kv(self):
        if self._kv_offload_mgr is not None:
            self._kv_offload_mgr.reset()
        from transformers.cache_utils import DynamicCache as _DC
        if self._decode_wrapper is not None:
            self._decode_wrapper.past_key_values = _DC()
        elif self._runtime is not None:
            self._runtime._reset_decode_cache()

    def run_one(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        with self._run_lock:
            return self._run_one_locked(prompt, output_len, input_len)

    def _run_one_locked(self, prompt: str, output_len: int, input_len: int) -> RequestResult:
        if self._runtime is None:
            return RequestResult(status="not_available", error_msg="not compiled")

        tok_out = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=input_len,
        )
        input_ids = tok_out["input_ids"]
        if self.batch_size > 1:
            input_ids = input_ids.expand(self.batch_size, -1).contiguous()

        self._reset_kv()
        _reset_peak()
        _sync()

        # Prefill (TTFT)
        t_pre0 = time.perf_counter()
        logits = self._runtime.prefill(input_ids.clone())
        _sync()
        t_pre1 = time.perf_counter()
        ttft = t_pre1 - t_pre0

        # Decode loop
        generated = []
        seq_len = input_ids.shape[1]
        t_dec0 = time.perf_counter()
        for step in range(output_len):
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_tok)
            logits = self._runtime.decode_step(next_tok, seq_len + step)
        _sync()
        t_dec1 = time.perf_counter()

        decode_s = t_dec1 - t_dec0
        out_toks  = self.batch_size * output_len

        return RequestResult(
            ttft_s=ttft,
            decode_s=decode_s,
            e2e_s=ttft + decode_s,
            input_tokens=input_ids.shape[1],
            output_tokens=out_toks,
            tok_per_s=out_toks / (ttft + decode_s) if (ttft + decode_s) > 0 else 0.0,
            decode_tok_per_s=out_toks / decode_s if decode_s > 0 else 0.0,
            peak_gpu_mb=_peak_gpu_mb(),
            peak_cpu_mb=_cpu_mb(),
        )

    def teardown(self):
        del self._runtime, self._model, self._kv_offload_mgr
        self._runtime = self._model = self._kv_offload_mgr = None
        # _decode_wrapper holds a direct reference to model CUDA layers and must
        # be explicitly cleared; forgetting this keeps 7+ GB pinned after teardown.
        if hasattr(self, '_decode_wrapper'):
            self._decode_wrapper = None
        # Clear ablation env vars so they don't bleed into subsequent conditions.
        os.environ.pop("HELM_DISABLE_AVX", None)
        os.environ.pop("HELM_DISABLE_ASYNC_KV", None)
        _free_memory()


# ─── Throughput sweep ─────────────────────────────────────────────────────────

def throughput_sweep(
    backend: Backend,
    prompts: List[str],
    output_len: int,
    input_len: int,
    concurrency_levels: List[int],
) -> Dict[int, Dict]:
    """
    Measure throughput at different concurrency levels.

    For each level C, send C requests as close to simultaneously as possible
    using a thread pool, then record wall-clock time and aggregate tok/s.
    """
    results = {}
    for c in concurrency_levels:
        batch = prompts[:c]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as pool:
            futs = [pool.submit(backend.run_one, p, output_len, input_len)
                    for p in batch]
            reqs = [f.result() for f in as_completed(futs)]
        t1 = time.perf_counter()

        wall_s    = t1 - t0
        ok        = [r for r in reqs if r.status == "success"]
        total_tok = sum(r.output_tokens for r in ok)
        results[c] = {
            "concurrency":     c,
            "wall_s":          wall_s,
            "n_success":       len(ok),
            "req_per_s":       len(ok) / wall_s if wall_s > 0 else 0.0,
            "total_tok_per_s": total_tok / wall_s if wall_s > 0 else 0.0,
            "mean_e2e_ms":     statistics.mean(r.e2e_s for r in ok) * 1000 if ok else 0.0,
        }
        print(f"  concurrency={c}: {results[c]['req_per_s']:.2f} req/s, "
              f"{results[c]['total_tok_per_s']:.1f} tok/s")
    return results


# ─── LM quality benchmarks ───────────────────────────────────────────────────

def run_lm_eval(
    model_name: str,
    tasks: List[str],
    dtype_str: str,
    output_dir: Path,
    limit: Optional[int] = 200,
) -> Dict:
    """
    Run lm-evaluation-harness on the specified tasks.
    Returns dict with per-task accuracy metrics.
    """
    try:
        import lm_eval
        from lm_eval import evaluator, tasks as lm_tasks
    except ImportError:
        print("[lm-eval] lm-evaluation-harness not installed — skipping quality benchmarks")
        print("  Install: uv pip install lm-eval")
        return {"error": "lm-eval not installed"}

    print(f"\n[lm-eval] Running: {tasks}  (limit={limit})")
    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},dtype={dtype_str}",
            tasks=tasks,
            num_fewshot=None,  # use default per task
            limit=limit,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size="auto",
            log_samples=False,
        )
        # Extract summary metrics
        summary = {}
        for task, task_res in results["results"].items():
            summary[task] = {k: v for k, v in task_res.items()
                             if isinstance(v, (int, float))}
            print(f"  {task}: {task_res}")
        return summary
    except Exception as e:
        print(f"[lm-eval] Failed: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


# ─── Feasibility probe ───────────────────────────────────────────────────────

def probe_feasibility(backend_name: str, args) -> Dict:
    """
    Try to load the model for one backend and report whether it fits in memory.

    Returns a dict with:
      fits_in_memory   bool   — True if setup() succeeded
      peak_gpu_mb      float  — peak GPU VRAM used after load (0 on CPU-only)
      peak_cpu_mb      float  — RSS after load (MB)
      gpu_free_mb      float  — GPU VRAM free after load (useful for KV headroom)
      status           str    — "ok" | "oom" | "error"
      error_msg        str
    """
    result: Dict = {
        "backend":        backend_name,
        "fits_in_memory": False,
        "peak_gpu_mb":    0.0,
        "peak_cpu_mb":    0.0,
        "gpu_free_mb":    0.0,
        "status":         "error",
        "error_msg":      "",
    }

    _reset_peak()
    _free_memory()

    backend = _make_backend(backend_name, args)
    try:
        ok = backend.setup()
        if ok:
            result["fits_in_memory"] = True
            result["peak_gpu_mb"]    = _peak_gpu_mb()
            result["peak_cpu_mb"]    = _cpu_mb()
            result["status"]         = "ok"
            if torch.cuda.is_available():
                free_bytes = torch.cuda.mem_get_info(0)[0]
                result["gpu_free_mb"] = free_bytes / 1024 ** 2
            print(f"  [{backend_name}] FITS — GPU={result['peak_gpu_mb']:.0f}MB  "
                  f"free={result['gpu_free_mb']:.0f}MB  CPU_RSS={result['peak_cpu_mb']:.0f}MB")
        else:
            result["status"]    = "error"
            result["error_msg"] = "setup() returned False"
            print(f"  [{backend_name}] FAILED (setup returned False)")
    except RuntimeError as e:
        msg = str(e)
        result["status"]    = "oom" if "out of memory" in msg.lower() else "error"
        result["error_msg"] = msg[:300]
        print(f"  [{backend_name}] OOM — {msg[:120]}")
    except Exception as e:
        result["status"]    = "error"
        result["error_msg"] = str(e)[:300]
        print(f"  [{backend_name}] ERROR — {str(e)[:120]}")
    finally:
        try:
            backend.teardown()
        except Exception:
            pass
        _free_memory()

    return result


# ─── Max decode length probe ──────────────────────────────────────────────────

def _run_one_probe(backend, prompt, cand, input_len, timeout_s) -> tuple:
    """Run a single probe candidate. Returns (RequestResult, entry dict)."""
    print(f"    output_len={cand} …", end=" ", flush=True)
    _set_alarm(timeout_s)
    try:
        r = backend.run_one(prompt, cand, input_len)
        _cancel_alarm()
    except _BenchTimeout as e:
        _cancel_alarm()
        r = RequestResult(status="timeout", error_msg=str(e))
    except RuntimeError as e:
        _cancel_alarm()
        msg = str(e)
        status = "oom" if "out of memory" in msg.lower() else "error"
        r = RequestResult(status=status, error_msg=msg[:200])
    except Exception as e:
        _cancel_alarm()
        r = RequestResult(status="error", error_msg=str(e)[:200])

    entry: Dict = {
        "output_len":        cand,
        "status":            r.status,
        "ttft_ms":           r.ttft_s * 1000,
        "decode_tok_per_s":  r.decode_tok_per_s,
        "e2e_s":             r.e2e_s,
        "peak_gpu_mb":       r.peak_gpu_mb,
        "peak_cpu_mb":       r.peak_cpu_mb,
        "error_msg":         r.error_msg,
    }

    if r.status == "success":
        print(f"OK  ({r.e2e_s:.1f}s, TTFT={r.ttft_s*1000:.0f}ms, "
              f"{r.decode_tok_per_s:.1f} tok/s, GPU={r.peak_gpu_mb:.0f}MB)")
    else:
        print(f"FAILED ({r.status}) — {r.error_msg[:80]}")

    return r, entry


def probe_max_decode_length(
    backend:    Backend,
    prompt:     str,
    input_len:  int,
    candidates: List[int],
    timeout_s:  int = 120,
) -> Dict:
    """
    Binary search over candidates to find the longest output length that succeeds.

    OOM/timeout is monotone: if length L fails, all L' > L also fail.
    Binary search finds the boundary in O(log N) probes instead of O(N).

    Returns:
      max_output_len  int   — largest successful output length (0 = all failed)
      results         list  — per-length dicts for every probe attempted
      oom_at          int   — first length that triggered OOM (None if never)
    """
    candidates = sorted(candidates)
    print(f"  [{backend.name}] max-decode probe (binary search): {candidates}")

    per_len: List[Dict] = []
    oom_at = None

    lo, hi = 0, len(candidates) - 1
    best_ok = -1  # index of highest confirmed success

    while lo <= hi:
        mid = (lo + hi) // 2
        cand = candidates[mid]

        r, entry = _run_one_probe(backend, prompt, cand, input_len, timeout_s)
        per_len.append(entry)

        if r.status == "success":
            best_ok = mid
            lo = mid + 1  # try longer
        else:
            if r.status == "oom" and oom_at is None:
                oom_at = cand
            hi = mid - 1  # try shorter
            _free_memory()

        if r.status == "success":
            _free_memory()

    max_len = candidates[best_ok] if best_ok >= 0 else 0

    return {
        "max_output_len": max_len,
        "oom_at":         oom_at,
        "results":        sorted(per_len, key=lambda e: e["output_len"]),
    }


# ─── Main experiment runner ───────────────────────────────────────────────────

def _hw_info() -> Dict:
    info: Dict = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu_name"]            = torch.cuda.get_device_name(0)
        info["gpu_count"]           = torch.cuda.device_count()
        info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    try:
        import psutil, platform
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"]  = psutil.cpu_count(logical=True)
        info["ram_total_gb"]       = psutil.virtual_memory().total / 1024 ** 3
        info["platform"]           = platform.platform()
    except Exception:
        pass
    try:
        import cpuinfo
        info["cpu_brand"] = cpuinfo.get_cpu_info().get("brand_raw", "unknown")
    except Exception:
        pass
    return info


def _build_prompts(n: int, base_prompt: str) -> List[str]:
    """Generate N slightly varied prompts for statistical measurement."""
    seeds = [
        "Explain the key differences between",
        "Describe the historical significance of",
        "What are the main advantages and disadvantages of",
        "Provide a detailed overview of",
        "Compare and contrast the approaches to",
        "Summarise the core principles behind",
        "What is the relationship between",
        "How does one typically approach",
        "Describe in detail the process of",
        "What are the most important aspects of",
    ]
    prompts = []
    for i in range(n):
        seed = seeds[i % len(seeds)]
        prompts.append(f"{seed} {base_prompt}")
    return prompts


def run_experiment(
    backend:       Backend,
    prompts:       List[str],
    output_len:    int,
    input_len:     int,
    ablation_tag:  str = "",
    timeout_s:     int = 300,
) -> BenchStats:
    """Run N requests through backend and return aggregated stats."""
    results = backend.run_n(prompts, output_len, input_len, timeout_s=timeout_s)
    stats = _aggregate(results, backend.name, ablation_tag)

    # Attach HELM compiler metrics if available
    if isinstance(backend, HelmBackend):
        stats.compile_time_s        = backend.compile_time_s
        stats.stage_plan            = backend.stage_plan
        stats.cost_model_decode_ms  = backend.cost_model_decode_ms
        stats.cost_model_prefill_ms = backend.cost_model_prefill_ms

    return stats


def _make_backend(name: str, args) -> Backend:
    if name == "vllm":
        return VLLMBackend(args.model, args.dtype,
                           gpu_memory_utilization=args.vllm_gpu_util)
    if name == "accelerate":
        return AccelerateBackend(args.model, args.dtype)
    if name == "deepspeed":
        return DeepSpeedBackend(args.model, args.dtype, cpu_offload=True)
    if name == "helm":
        return HelmBackend(
            args.model, args.dtype,
            kv_offload=not args.no_kv_offload,
            cpu_threads=args.cpu_threads,
            input_len=args.input_len,
            batch_size=args.batch_size,
        )
    raise ValueError(f"Unknown backend: {name}")


def _incremental_save(results: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def run_all(args) -> Dict:
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "paper_results.json"

    results: Dict[str, Any] = {
        "config":            vars(args),
        "hardware":          _hw_info(),
        "feasibility":       {},
        "max_decode_length": {},
        "latency_sweep":     {},
        "throughput":        {},
        "ablations":         {},
        "lm_eval":           {},
    }
    _incremental_save(results, json_path)

    output_lens: List[int] = sorted(set(args.output_lens))
    prompts = _build_prompts(args.num_requests, args.base_prompt)

    # ── 0. Feasibility probe ──────────────────────────────────────────────────
    if not args.skip_feasibility:
        print(f"\n{'='*70}")
        print(f"  SECTION 0: Feasibility Probe  backends={args.backends}")
        print(f"  (Does the model load? What is the memory footprint?)")
        print(f"{'='*70}")
        for bname in args.backends:
            print(f"\n  [{bname}] loading model …")
            results["feasibility"][bname] = probe_feasibility(bname, args)
            _incremental_save(results, json_path)

    # ── 0b. Max decode length probe ───────────────────────────────────────────
    if not args.skip_max_decode:
        print(f"\n{'='*70}")
        print(f"  SECTION 0b: Max Decode Length  candidates={args.max_decode_candidates}")
        print(f"  (Longest sequence each backend can produce before OOM)")
        print(f"{'='*70}")
        probe_prompt = prompts[0]
        for bname in args.backends:
            print(f"\n  ── {bname.upper()} ──")
            backend = _make_backend(bname, args)
            ok = backend.setup()
            if not ok:
                results["max_decode_length"][bname] = {
                    "max_output_len": 0, "oom_at": None,
                    "results": [], "error": "setup failed",
                }
                _incremental_save(results, json_path)
                continue
            results["max_decode_length"][bname] = probe_max_decode_length(
                backend, probe_prompt, args.input_len,
                args.max_decode_candidates, timeout_s=args.timeout,
            )
            backend.teardown()
            _free_memory()
            _incremental_save(results, json_path)

        # Print comparison table
        print(f"\n  ── Max decode length summary ──")
        for bname, mres in results["max_decode_length"].items():
            max_l = mres.get("max_output_len", 0)
            oom   = mres.get("oom_at", "—")
            print(f"    {bname:15s}: max={max_l:6d}  oom_at={oom}")

    # ── 1. Latency sweep (all backends × all output lengths) ─────────────────
    if args.skip_latency_sweep:
        print("  [skip] Latency sweep disabled via --skip-latency-sweep")
    else:
      print(f"\n{'='*70}")
      print(f"  SECTION 1: Latency Sweep  backends={args.backends}  "
            f"output_lens={output_lens}  N={args.num_requests}")
      print(f"{'='*70}")

    for bname in args.backends if not args.skip_latency_sweep else []:
        print(f"\n{'─'*60}")
        print(f"  Backend: {bname.upper()}")
        print(f"{'─'*60}")
        backend = _make_backend(bname, args)
        ok = backend.setup()
        if not ok:
            results["latency_sweep"][bname] = {"error": "setup failed"}
            _incremental_save(results, json_path)
            continue

        if args.num_warmup > 0:
            print(f"  [{bname}] Warming up ({args.num_warmup} request(s)) …")
            for _ in range(args.num_warmup):
                try:
                    backend.run_one(prompts[0], output_lens[0], args.input_len)
                except Exception:
                    pass

        results["latency_sweep"][bname] = {}
        for ol in output_lens:
            print(f"\n  output_len={ol} …")
            stats = run_experiment(
                backend, prompts[:args.num_requests], ol,
                args.input_len, ablation_tag=f"out={ol}",
                timeout_s=args.timeout,
            )
            d = asdict(stats)
            d.pop("raw_requests", None)   # keep JSON compact
            results["latency_sweep"][bname][str(ol)] = d
            _incremental_save(results, json_path)
            _print_stats(stats, ol)

        # Throughput sweep for this backend
        if args.throughput_concurrency:
            print(f"\n  [{bname}] Throughput sweep: {args.throughput_concurrency}")
            try:
                results["throughput"][bname] = throughput_sweep(
                    backend, prompts, output_lens[0],
                    args.input_len, args.throughput_concurrency)
            except Exception as exc:
                print(f"  [{bname}] Throughput sweep failed: {exc}")
                results["throughput"][bname] = {"error": str(exc)}
            _incremental_save(results, json_path)

        backend.teardown()
        _free_memory()

    # ── 2. Ablations (HELM only) ──────────────────────────────────────────────
    if "helm" in args.backends and args.ablations:
        print(f"\n{'='*70}")
        print(f"  SECTION 2: Ablations  {args.ablations}")
        print(f"{'='*70}")

        ablation_output_len = output_lens[len(output_lens) // 2]  # mid-range

        for ablation in args.ablations:
            print(f"\n  ── Ablation: {ablation} ──")
            try:
                ablation_results = _run_ablation(ablation, args, prompts,
                                                 ablation_output_len, args.input_len)
            except Exception as exc:
                import traceback
                print(f"  [ERROR] Ablation '{ablation}' crashed: {exc}")
                traceback.print_exc()
                ablation_results = {"error": str(exc)}
            results["ablations"][ablation] = ablation_results
            _incremental_save(results, json_path)

    # ── 3. LM quality benchmarks ──────────────────────────────────────────────
    if not args.no_lm_eval and args.lm_eval_tasks:
        print(f"\n{'='*70}")
        print(f"  SECTION 3: LM Quality Benchmarks  tasks={args.lm_eval_tasks}")
        print(f"{'='*70}")
        results["lm_eval"] = run_lm_eval(
            args.model, args.lm_eval_tasks, args.dtype,
            out_dir, limit=args.lm_eval_limit,
        )
        _incremental_save(results, json_path)

    print(f"\n{'='*70}")
    print(f"  All experiments done. Results: {json_path}")
    print(f"{'='*70}")
    return results


def _run_ablation(
    ablation:   str,
    args,
    prompts:    List[str],
    output_len: int,
    input_len:  int,
) -> Dict:
    """Run one ablation study and return results dict."""
    ab_results = {}

    if ablation == "batch_size":
        for bs in [1, 2, 4, 8]:
            tag = f"batch={bs}"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=args.cpu_threads,
                            input_len=input_len, batch_size=bs)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "context_length":
        for ctx in [128, 256, 512, 1024, 2048]:
            tag = f"ctx={ctx}"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=args.cpu_threads,
                            input_len=ctx, batch_size=args.batch_size)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       ctx, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "no_avx":
        for disable_avx in [False, True]:
            tag = "with_avx" if not disable_avx else "no_avx"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=args.cpu_threads,
                            input_len=input_len, batch_size=args.batch_size,
                            disable_avx=disable_avx)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "no_kv_offload":
        for kv_off in [True, False]:
            tag = "kv_offload" if kv_off else "no_kv_offload"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=kv_off,
                            cpu_threads=args.cpu_threads,
                            input_len=input_len, batch_size=args.batch_size)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "cpu_threads":
        for threads in [1, 2, 4, 8, 16]:
            tag = f"threads={threads}"
            print(f"  {tag} …")
            b = HelmBackend(args.model, args.dtype,
                            kv_offload=not args.no_kv_offload,
                            cpu_threads=threads,
                            input_len=input_len, batch_size=args.batch_size)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    elif ablation == "vs_baselines":
        # Direct comparison at fixed output_len: all 4 backends side by side
        for bname in ["accelerate", "deepspeed", "vllm", "helm"]:
            tag = bname
            print(f"  {tag} …")
            b = _make_backend(bname, args)
            if b.setup():
                stats = run_experiment(b, prompts[:args.num_requests], output_len,
                                       input_len, ablation_tag=tag, timeout_s=args.timeout)
                d = asdict(stats); d.pop("raw_requests", None)
                ab_results[tag] = d
                _print_stats(stats, output_len)
            else:
                ab_results[tag] = {"error": "setup failed"}
            b.teardown(); _free_memory()

    else:
        print(f"  Unknown ablation: {ablation}")

    return ab_results


def _print_stats(stats: BenchStats, output_len: int):
    ok_frac = f"{stats.n_success}/{stats.n_requests}"
    print(f"  [{stats.backend}] output={output_len}  ({ok_frac} succeeded)")
    if stats.n_success == 0:
        return
    print(f"    TTFT    : p50={stats.ttft_p50:.1f}ms  p95={stats.ttft_p95:.1f}ms  "
          f"p99={stats.ttft_p99:.1f}ms  mean={stats.ttft_mean:.1f}ms")
    print(f"    Decode  : p50={stats.decode_lat_p50:.1f}ms/tok  "
          f"p95={stats.decode_lat_p95:.1f}ms/tok  "
          f"p99={stats.decode_lat_p99:.1f}ms/tok")
    print(f"    E2E     : p50={stats.e2e_p50:.0f}ms  p95={stats.e2e_p95:.0f}ms  "
          f"p99={stats.e2e_p99:.0f}ms")
    print(f"    Tok/s   : {stats.tok_per_s_mean:.1f}  |  "
          f"decode tok/s: {stats.decode_tok_per_s_mean:.1f}")
    print(f"    Memory  : GPU={stats.peak_gpu_mb_mean:.0f}MB  CPU={stats.peak_cpu_mb_mean:.0f}MB")
    if stats.compile_time_s > 0:
        print(f"    Compiler: {stats.compile_time_s:.1f}s  plan=[{stats.stage_plan}]")
        print(f"    CostModel: decode={stats.cost_model_decode_ms:.1f}ms  "
              f"prefill={stats.cost_model_prefill_ms:.0f}ms")
        if stats.decode_lat_mean > 0:
            err_pct = abs(stats.cost_model_decode_ms - stats.decode_lat_mean) / stats.decode_lat_mean * 100
            print(f"    CostModel error: {err_pct:.1f}% vs measured {stats.decode_lat_mean:.1f}ms/tok")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model name or local path")
    p.add_argument("--backends", nargs="+",
                   choices=["helm", "accelerate", "vllm", "deepspeed"],
                   default=["vllm", "accelerate", "deepspeed", "helm"],
                   help="Which backends to benchmark")
    p.add_argument("--base-prompt", default=(
        "transformer-based language models and their applications in natural language processing"),
                   help="Base text appended to each varied prompt")
    p.add_argument("--input-len", type=int, default=64,
                   help="Max input tokens (truncates prompt if needed)")
    p.add_argument("--output-lens", type=int, nargs="+",
                   default=[64, 128, 256, 512],
                   help="Output lengths for the latency sweep")
    p.add_argument("--num-requests", type=int, default=20,
                   help="Number of independent requests for statistical measurement")
    p.add_argument("--num-warmup", type=int, default=1,
                   help="Warmup requests before each timed section (discarded)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Request batch size (HELM and Accelerate)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--timeout", type=int, default=3600,
                   help="Per-request wall-clock timeout in seconds")

    # Backend-specific
    p.add_argument("--no-kv-offload", action="store_true",
                   help="Disable KV offload in HELM")
    p.add_argument("--cpu-threads", type=int, default=8,
                   help="OMP/MKL thread count for CPU stages")
    p.add_argument("--vllm-gpu-util", type=float, default=0.90)

    # Throughput
    p.add_argument("--throughput-concurrency", type=int, nargs="*",
                   default=[1, 2, 4, 8],
                   help="Concurrency levels for throughput sweep (0 to skip)")

    # Ablations
    p.add_argument("--ablations", nargs="*",
                   choices=["batch_size", "context_length", "no_avx",
                            "no_kv_offload", "cpu_threads", "vs_baselines"],
                   default=[],
                   help="Ablation studies to run (all HELM unless noted)")

    # Section skip flags (used by the orchestration script to isolate one section per subprocess)
    p.add_argument("--skip-latency-sweep", action="store_true",
                   help="Skip the latency sweep (Section 1)")
    p.add_argument("--skip-feasibility", action="store_true",
                   help="Skip the memory feasibility probe (Section 0)")
    p.add_argument("--skip-max-decode", action="store_true",
                   help="Skip the max decode length probe (Section 0b)")
    p.add_argument("--max-decode-candidates", type=int, nargs="+",
                   default=[128, 512, 1024, 2048,
                            3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
                            11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432,
                            19456, 20480, 21504, 22528, 23552, 24576, 25600, 26624,
                            27648, 28672, 29696, 30720, 31744, 32768],
                   help="Output lengths to try in the max-decode probe (stops at first OOM)")

    # LM quality
    p.add_argument("--no-lm-eval", action="store_true", default=True,
                   help="Skip lm-evaluation-harness quality benchmarks (default: True)")
    p.add_argument("--lm-eval-tasks", nargs="+",
                   default=["mmlu", "hellaswag", "arc_easy"],
                   help="lm-eval tasks to evaluate")
    p.add_argument("--lm-eval-limit", type=int, default=200,
                   help="Max samples per task (None = all; use small value for quick checks)")

    # Output
    p.add_argument("--output-dir", default="experiments/results",
                   help="Directory to write JSON results")

    return p.parse_args()


def main():
    args = _parse_args()

    print(f"\n{'#'*70}")
    print(f"  HELM Paper Experiments")
    print(f"  Model      : {args.model}")
    print(f"  Backends   : {args.backends}")
    print(f"  Output lens: {args.output_lens}")
    print(f"  N requests : {args.num_requests}")
    print(f"  Feasibility: {'skip' if args.skip_feasibility else 'yes'}")
    print(f"  Max decode : {'skip' if args.skip_max_decode else str(args.max_decode_candidates)}")
    print(f"  Ablations  : {args.ablations or 'none'}")
    print(f"  LM eval    : {'disabled' if args.no_lm_eval else args.lm_eval_tasks}")
    hw = _hw_info()
    print(f"  Hardware   : {hw.get('cpu_brand','?')}  "
          f"RAM={hw.get('ram_total_gb',0):.0f}GB  "
          f"GPU={hw.get('gpu_name','none')}")
    print(f"{'#'*70}\n")

    run_all(args)


if __name__ == "__main__":
    main()
