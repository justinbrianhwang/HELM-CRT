from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import List

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helm",
        description="HELM CLI for compiler planning and runtime execution.",
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="plan", choices=["baseline", "import", "units", "plan", "lower", "execute_stagewise", "execute_full", "dry_run"])
    parser.add_argument("--prompt", type=str, default="Explain what a compiler does.")
    parser.add_argument("--max-input-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    compiler = parser.add_argument_group("Compiler Flags")
    compiler.add_argument("--compiler-plan", type=str, default="auto", choices=["auto", "manual"])
    compiler.add_argument("--compiler-cpu-layers", type=str, help="Manual plan CPU layer range, e.g. 0:13")
    compiler.add_argument("--compiler-gpu-layers", type=str, help="Manual plan GPU layer range, e.g. 14:27")
    compiler.add_argument("--print-graph-summary", action="store_true")
    compiler.add_argument("--print-partition-units", action="store_true")
    compiler.add_argument("--print-plan", action="store_true")

    runtime = parser.add_argument_group("Runtime Flags")
    runtime.add_argument("--runtime-skip-baseline", action="store_true")
    runtime.add_argument("--runtime-baseline-on-cpu", action="store_true")
    runtime.add_argument("--kv-offload", action="store_true",
                         help="Enable paged KV cache with CPU offloading for long-context generation")

    infra = parser.add_argument_group("Backend Flags")
    infra.add_argument("--load-device", type=str, default="cpu", choices=["cpu", "cuda"])
    infra.add_argument("--trace-device", type=str, default="cpu", choices=["cpu", "cuda"])
    infra.add_argument("--execution-device", type=str, default="cpu", choices=["cpu", "cuda"])
    infra.add_argument("--cpu-threads", type=int, default=6)
    infra.add_argument("--save-artifacts-dir", type=str, default="artifacts")

    return parser


def _to_benchmark_argv(args: argparse.Namespace) -> List[str]:
    argv: List[str] = [
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--plan",
        args.compiler_plan,
        "--load-device",
        args.load_device,
        "--trace-device",
        args.trace_device,
        "--execution-device",
        args.execution_device,
        "--cpu-threads",
        str(args.cpu_threads),
        "--save-artifacts-dir",
        args.save_artifacts_dir,
    ]

    if args.compiler_cpu_layers:
        argv.extend(["--cpu-layers", args.compiler_cpu_layers])
    if args.compiler_gpu_layers:
        argv.extend(["--gpu-layers", args.compiler_gpu_layers])
    if args.print_graph_summary:
        argv.append("--print-graph-summary")
    if args.print_partition_units:
        argv.append("--print-partition-units")
    if args.print_plan:
        argv.append("--print-plan")
    if args.runtime_skip_baseline:
        argv.append("--skip-baseline")
    if args.runtime_baseline_on_cpu:
        argv.append("--baseline-on-cpu")
    if args.kv_offload:
        argv.append("--kv-offload")

    return argv


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    root_dir = Path(__file__).resolve().parents[1]
    benchmark_path = root_dir / "benchmarks" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("helm_benchmark_main", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark entrypoint from {benchmark_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(_to_benchmark_argv(args))


if __name__ == "__main__":
    main()
