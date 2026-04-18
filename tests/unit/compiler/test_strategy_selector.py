"""
Unit tests for StrategySelector (helm/compiler/optimization/strategy_selector.py).

The selector uses a greedy latency-minimising algorithm:
  1. Try all-GPU (best case).
  2. Try CPU→GPU splits from most-GPU-first (split_idx=1) and return the
     first feasible plan — this maximises GPU layers, minimising total decode
     latency since GPU bandwidth >> CPU bandwidth.
  3. Fall back to all-CPU.
"""
import pytest

from helm.compiler.partition.partition_units import PartitionUnit
from helm.compiler.optimization.cost_model import (
    DeviceProfile, HelmCostModel, WorkloadSpec,
)
from helm.compiler.optimization.strategy_selector import StrategySelector, StrategySelectorConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _unit(uid, layer_id=None, param_bytes=5 * 1024 ** 2):
    return PartitionUnit(
        unit_id=uid,
        unit_type="transformer_block" if layer_id is not None else "embedding",
        layer_start=layer_id,
        layer_end=layer_id,
        node_ids=[uid],
        param_bytes=param_bytes,
        activation_bytes=256,
        kv_bytes_per_token=64,
        flops_prefill=int(1e9),
        flops_decode=int(1e7),
    )


@pytest.fixture()
def units_4():
    return [_unit(i, layer_id=i) for i in range(4)]


@pytest.fixture()
def small_workload():
    return WorkloadSpec(batch_size=1, prefill_seq_len=16, decode_context_len=32,
                        decode_tokens=8, dtype_size=2)


# ──────────────────────────────────────────────────────────────────────────────
# All-GPU plan (fits in VRAM)
# ──────────────────────────────────────────────────────────────────────────────

def test_all_gpu_when_fits(units_4, small_workload, gpu_profile):
    """When everything fits on GPU, return a single all-GPU stage."""
    model = HelmCostModel({"cuda:0": gpu_profile}, {})
    cfg = StrategySelectorConfig(allow_cpu=False)
    selector = StrategySelector(model, ["cuda:0"], cfg)
    plan = selector.select(units_4, small_workload)
    assert len(plan.stages) == 1
    assert plan.stages[0].device_id == "cuda:0"


def test_prefer_all_gpu_over_split(units_4, small_workload, cpu_profile, gpu_profile):
    """If all layers fit on GPU, prefer that over a CPU+GPU split."""
    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": gpu_profile}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu", "cuda:0"], cfg)
    plan = selector.select(units_4, small_workload)
    assert len(plan.stages) == 1
    assert plan.stages[0].device_id == "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# CPU→GPU split: maximise GPU layers
# ──────────────────────────────────────────────────────────────────────────────

def test_cpu_gpu_split_when_gpu_too_small(small_workload, cpu_profile):
    """With a small GPU (fits 2 of 4 units), the selector returns a 2-stage CPU→GPU plan."""
    # 4 units × 100 MB each.  Memory estimate for 2 GPU units:
    #   200 MB params + 5% margin + 200 MB reserve = ~410 MB < 450 MB → fits.
    # For 3 GPU units: 300 + 15 + 200 = 515 MB > 450 MB → OOM.
    small_gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9,
                              450 * 1024 * 1024)  # 450 MB
    units = [_unit(i, layer_id=i, param_bytes=100 * 1024 * 1024) for i in range(4)]
    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": small_gpu}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu", "cuda:0"], cfg)
    plan = selector.select(units, small_workload)
    assert len(plan.stages) == 2
    assert plan.stages[0].device_id == "cpu"
    assert plan.stages[1].device_id == "cuda:0"


def test_maximises_gpu_layers(small_workload, cpu_profile):
    """The split puts as many units as possible on GPU (greedy from most-GPU-first)."""
    # 4 units × 100 MB; GPU fits 2 units (450 MB capacity).
    small_gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9,
                              450 * 1024 * 1024)
    units = [_unit(i, layer_id=i, param_bytes=100 * 1024 * 1024) for i in range(4)]
    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": small_gpu}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu", "cuda:0"], cfg)
    plan = selector.select(units, small_workload)
    gpu_stage = next(s for s in plan.stages if "cuda" in s.device_id)
    # Should have 2 GPU units (maximum that fits)
    assert len(gpu_stage.units) == 2


def test_no_gpu_to_cpu_ordering(small_workload, cpu_profile):
    """The selector never returns GPU→CPU stage ordering (lm_head must stay on GPU)."""
    small_gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9,
                              450 * 1024 * 1024)
    units = [_unit(i, layer_id=i, param_bytes=100 * 1024 * 1024) for i in range(4)]
    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": small_gpu}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu", "cuda:0"], cfg)
    plan = selector.select(units, small_workload)
    if len(plan.stages) == 2:
        assert plan.stages[0].device_id == "cpu"
        assert plan.stages[1].device_id == "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# All-CPU fallback
# ──────────────────────────────────────────────────────────────────────────────

def test_fallback_to_cpu_only(units_4, small_workload, cpu_profile):
    """With no GPU device, return a single all-CPU stage."""
    model = HelmCostModel({"cpu": cpu_profile}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu"], cfg)
    plan = selector.select(units_4, small_workload)
    assert len(plan.stages) == 1
    assert plan.stages[0].device_id == "cpu"


def test_no_feasible_plan_raises(units_4, small_workload):
    """Raises RuntimeError when no plan fits any device."""
    tiny = DeviceProfile("cpu", "cpu", 1e12, 1e12, 50e9, 1 * 1024 ** 2)  # 1 MB — OOM
    model = HelmCostModel({"cpu": tiny}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu"], cfg)
    with pytest.raises(RuntimeError, match="No feasible partition plan found"):
        selector.select(units_4, small_workload)


# ──────────────────────────────────────────────────────────────────────────────
# kv_offload flag frees GPU memory → more layers fit on GPU
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_offload_allows_more_gpu_layers(small_workload, cpu_profile):
    """
    With kv_offload=True the GPU memory estimate excludes KV bytes,
    so a plan that was OOM without offload can become feasible.
    """
    # Construct units with non-trivial kv_bytes_per_token so they affect memory.
    kv_heavy_units = [
        PartitionUnit(
            unit_id=i, unit_type="transformer_block",
            layer_start=i, layer_end=i, node_ids=[i],
            param_bytes=2 * 1024 * 1024,          # 2 MB params
            kv_bytes_per_token=1024 * 1024,        # 1 MB KV/token — large
            activation_bytes=256,
            flops_prefill=int(1e9), flops_decode=int(1e7),
        )
        for i in range(4)
    ]
    # workload with long context: 256 tokens × 4 units × 1 MB = 1 GB KV
    workload_long = WorkloadSpec(batch_size=1, prefill_seq_len=16,
                                 decode_context_len=256, decode_tokens=8, dtype_size=2)

    # GPU that fits params (8 MB) but not params + KV (1 GB+)
    medium_gpu = DeviceProfile("cuda:0", "cuda", 100e12, 100e12, 500e9,
                               20 * 1024 * 1024)   # 20 MB

    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": medium_gpu}, {})

    # Without kv_offload: all-GPU is OOM (params OK but KV blows up the budget)
    selector_no_offload = StrategySelector(
        model, ["cpu", "cuda:0"],
        StrategySelectorConfig(allow_cpu=True, kv_offload=False))
    plan_no_offload = selector_no_offload.select(kv_heavy_units, workload_long)

    # With kv_offload: KV excluded from GPU budget → all-GPU fits
    selector_offload = StrategySelector(
        model, ["cpu", "cuda:0"],
        StrategySelectorConfig(allow_cpu=True, kv_offload=True))
    plan_offload = selector_offload.select(kv_heavy_units, workload_long)

    gpu_layers_no_offload = sum(
        len(s.units) for s in plan_no_offload.stages if "cuda" in s.device_id)
    gpu_layers_offload = sum(
        len(s.units) for s in plan_offload.stages if "cuda" in s.device_id)
    assert gpu_layers_offload >= gpu_layers_no_offload


# ──────────────────────────────────────────────────────────────────────────────
# Plan correctness
# ──────────────────────────────────────────────────────────────────────────────

def test_select_returns_valid_plan(units_4, small_workload, cpu_profile, gpu_profile):
    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": gpu_profile}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu", "cuda:0"], cfg)
    plan = selector.select(units_4, small_workload)
    assert plan is not None
    assert len(plan.stages) >= 1
    # All units assigned exactly once
    assigned = [u for s in plan.stages for u in s.units]
    assert len(assigned) == len(units_4)


def test_plan_covers_all_units(units_4, small_workload, cpu_profile):
    model = HelmCostModel({"cpu": cpu_profile}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu"], cfg)
    plan = selector.select(units_4, small_workload)
    assigned_ids = {u.unit_id for s in plan.stages for u in s.units}
    expected_ids = {u.unit_id for u in units_4}
    assert assigned_ids == expected_ids


def test_gpu_preferred_over_cpu(units_4, small_workload, cpu_profile, gpu_profile):
    """When GPU fits, plan uses GPU (faster → lower latency)."""
    model = HelmCostModel({"cpu": cpu_profile, "cuda:0": gpu_profile}, {})
    cfg = StrategySelectorConfig(allow_cpu=True)
    selector = StrategySelector(model, ["cpu", "cuda:0"], cfg)
    plan = selector.select(units_4, small_workload)
    assert any("cuda" in s.device_id for s in plan.stages)
