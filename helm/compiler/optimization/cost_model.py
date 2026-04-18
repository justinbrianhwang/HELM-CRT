# helm/compiler/optimization/cost_model.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from helm.compiler.partition.parition_plan import PartitionPlan, StageSpec
from helm.compiler.partition.partition_units import PartitionUnit


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Architecture constants extracted from model.config.
    Passed to HelmCostModel so it can compute analytical FLOPs and memory
    traffic without relying on the pre-baked estimates stored in PartitionUnit
    (those embed the analysis sequence length and cannot be rescaled reliably).
    """
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int           # = hidden_size // num_attention_heads
    dtype_size: int = 2     # bytes per element (2 = fp16/bf16, 4 = fp32)

    @staticmethod
    def from_hf_config(cfg, dtype_size: int = 2) -> "ModelConfig":
        num_heads    = int(getattr(cfg, "num_attention_heads", 1))
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
        hidden_size  = int(getattr(cfg, "hidden_size", 0))
        head_dim     = int(getattr(cfg, "head_dim",
                           hidden_size // num_heads if num_heads > 0 else 0))
        return ModelConfig(
            hidden_size       = hidden_size,
            intermediate_size = int(getattr(cfg, "intermediate_size", 0)),
            num_attention_heads = num_heads,
            num_kv_heads      = num_kv_heads,
            head_dim          = head_dim,
            dtype_size        = dtype_size,
        )


@dataclass
class DeviceProfile:
    device_id: str
    device_type: str            # "cpu" | "cuda"
    peak_flops_prefill: float   # measured effective FLOPS/s (from device_profiler microbenchmark)
    peak_flops_decode: float    # measured effective FLOPS/s at seq=1 (GEMV regime)
    mem_bandwidth: float        # measured DRAM bandwidth (bytes/s, from stream copy benchmark)
    memory_capacity: int        # bytes
    # ── calibration factors ──────────────────────────────────────────────────
    # These are multiplicative efficiency factors (0 < eff ≤ 1.0) applied on
    # top of the microbenchmark-measured peak values.  Default 1.0 means the
    # profiler's measurement is already representative (i.e. no additional
    # de-rating needed).  Set below 1.0 if validate() shows the model over-
    # predicts performance for the actual workload shapes.
    efficiency_compute: float = 1.0
    efficiency_memory:  float = 1.0
    # ── CPU L3 cache model ───────────────────────────────────────────────────
    # When set, attention KV accesses that fit in L3 use l3_bandwidth instead
    # of mem_bandwidth.  0 disables L3 modeling (conservative; safe default).
    l3_size_bytes: int   = 0
    l3_bandwidth:  float = 0.0


@dataclass
class LinkProfile:
    src: str
    dst: str
    bandwidth_bytes_per_s: float
    latency_s: float


@dataclass
class WorkloadSpec:
    batch_size: int
    prefill_seq_len: int
    decode_context_len: int
    decode_tokens: int
    dtype_size: int


@dataclass
class StageCost:
    stage_id: int
    device: str
    param_bytes: int
    activation_bytes: int
    kv_bytes: int
    memory_bytes: int
    prefill_compute_s: float
    prefill_memory_s: float
    prefill_comm_s: float
    prefill_total_s: float
    decode_compute_s: float
    decode_memory_s: float
    decode_comm_s: float
    decode_total_s: float
    feasible: bool
    # Whether the decode step is compute- or memory-bandwidth-bound
    decode_regime: str = "memory"   # "memory" | "compute"
    reason: Optional[str] = None


@dataclass
class PlanCost:
    feasible: bool
    stage_costs: List[StageCost] = field(default_factory=list)
    prefill_latency_s: float = 0.0
    decode_token_latency_s: float = 0.0
    total_latency_s: float = 0.0
    throughput_tokens_per_s: float = 0.0
    max_stage_memory_bytes: int = 0
    reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

class HelmCostModel:
    """
    Roofline-based cost model for heterogeneous CPU+GPU inference.

    Design principles
    -----------------
    All FLOPs and memory-traffic formulas are derived analytically from
    ModelConfig so they scale correctly with the actual workload (batch size,
    context length) rather than relying on the analysis-time estimates
    embedded in PartitionUnit (which have B and S baked in at analysis time
    and cannot be correctly rescaled for arbitrary workloads).

    param_bytes and kv_bytes_per_token from PartitionUnit are used directly —
    they are workload-independent and measured exactly from the model weights.

    Roofline: stage_time = max(compute_time, memory_time).

    At decode (seq=1) projection layers are always memory-bandwidth-bound on
    both CPU and GPU for current hardware.  At large prefill sequences the GPU
    can flip to compute-bound for projections once arithmetic intensity
    (FLOPs/byte) exceeds the hardware ops-per-byte ratio.
    """

    def __init__(
        self,
        devices: Dict[str, DeviceProfile],
        links: Dict[Tuple[str, str], LinkProfile],
        model_config: Optional[ModelConfig] = None,
    ):
        self.devices      = devices
        self.links        = links
        self.model_config = model_config   # None → falls back to unit FLOPs

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate_plan(
        self,
        plan: PartitionPlan,
        workload: WorkloadSpec,
        kv_offload: bool = False,
        kv_reserve_tokens: int = 0,
    ) -> PlanCost:
        plan_cost = PlanCost(feasible=True)
        max_stage_mem   = 0
        total_prefill_s = 0.0
        total_decode_s  = 0.0

        for i, stage in enumerate(plan.stages):
            next_stage = plan.stages[i + 1] if i + 1 < len(plan.stages) else None
            sc = self.estimate_stage(stage, workload, next_stage, kv_offload=kv_offload,
                                     kv_reserve_tokens=kv_reserve_tokens)
            plan_cost.stage_costs.append(sc)

            if not sc.feasible:
                plan_cost.feasible = False
                plan_cost.reason = (
                    f"Stage {stage.stage_id} on {stage.device_id} infeasible: {sc.reason}"
                )
                return plan_cost

            max_stage_mem   = max(max_stage_mem, sc.memory_bytes)
            total_prefill_s += sc.prefill_total_s
            total_decode_s  += sc.decode_total_s

        plan_cost.prefill_latency_s      = total_prefill_s
        plan_cost.decode_token_latency_s = total_decode_s
        plan_cost.total_latency_s = (
            total_prefill_s + workload.decode_tokens * total_decode_s
        )
        if total_decode_s > 0:
            plan_cost.throughput_tokens_per_s = workload.batch_size / total_decode_s
        plan_cost.max_stage_memory_bytes = max_stage_mem
        return plan_cost

    def estimate_stage(
        self,
        stage: StageSpec,
        workload: WorkloadSpec,
        next_stage: Optional[StageSpec] = None,
        kv_offload: bool = False,
        kv_reserve_tokens: int = 0,
    ) -> StageCost:
        device = self.devices.get(stage.device_id)
        if not device:
            return StageCost(
                stage_id=stage.stage_id, device=stage.device_id,
                param_bytes=0, activation_bytes=0, kv_bytes=0, memory_bytes=0,
                prefill_compute_s=0, prefill_memory_s=0, prefill_comm_s=0,
                prefill_total_s=0, decode_compute_s=0, decode_memory_s=0,
                decode_comm_s=0, decode_total_s=0,
                feasible=False, reason=f"Unknown device: {stage.device_id}",
            )

        # Memory sizing
        param_bytes, act_prefill, act_decode, kv_bytes = self._stage_memory(
            stage, workload
        )
        if kv_offload and device.device_type == "cuda":
            # KV pages are evicted to CPU RAM as context grows, so the full kv_bytes
            # budget need not fit on GPU. However, reserve headroom for the first
            # kv_reserve_tokens tokens, which are always kept on GPU before eviction.
            kv_per_token = sum(u.kv_bytes_per_token for u in stage.units)
            kv_for_budget = kv_per_token * workload.batch_size * kv_reserve_tokens
        else:
            kv_for_budget = kv_bytes
        margin = int(0.05 * (param_bytes + act_prefill)) + (200 * 1024 * 1024)
        total_mem = param_bytes + max(act_prefill, act_decode) + kv_for_budget + margin

        feasible = total_mem <= device.memory_capacity
        reason   = (
            None if feasible
            else f"OOM: need {total_mem/1e9:.2f} GB, device has {device.memory_capacity/1e9:.2f} GB"
        )

        pre_comp, pre_mem, pre_base = self._prefill_time(stage, workload, device)
        dec_comp, dec_mem, dec_base = self._decode_time(stage, workload, device)
        pre_comm, dec_comm = self._boundary_comm(stage, next_stage, workload)

        return StageCost(
            stage_id         = stage.stage_id,
            device           = stage.device_id,
            param_bytes      = param_bytes,
            activation_bytes = max(act_prefill, act_decode),
            kv_bytes         = kv_bytes,
            memory_bytes     = total_mem,
            prefill_compute_s= pre_comp,
            prefill_memory_s = pre_mem,
            prefill_comm_s   = pre_comm,
            prefill_total_s  = pre_base + pre_comm,
            decode_compute_s = dec_comp,
            decode_memory_s  = dec_mem,
            decode_comm_s    = dec_comm,
            decode_total_s   = dec_base + dec_comm,
            feasible         = feasible,
            decode_regime    = "compute" if dec_comp > dec_mem else "memory",
            reason           = reason,
        )

    def validate(
        self,
        plan: PartitionPlan,
        workload: WorkloadSpec,
        measured_ms: Dict[int, Dict[str, float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Compare cost model predictions against measured per-stage wall times.

        Parameters
        ----------
        measured_ms
            {stage_id: {"decode_ms": float, "prefill_ms": float}}
            Actual measured times, e.g. from executor._PROFILE output.

        Returns
        -------
        Per-stage dict with predicted vs actual times, percent error, and
        suggested efficiency factors that would make the model match the
        measurements.  Pass these as DeviceProfile.efficiency_memory /
        efficiency_compute to calibrate the model for this hardware.

        How to interpret suggested_efficiency_memory
        --------------------------------------------
        If the stage is memory-bound, the raw memory time (bytes/bw) is
        `sc.decode_memory_s`.  The model predicts `raw / efficiency`.  If
        the prediction is off, the implied efficiency that would match the
        measurement is `raw / actual_s`.  A value < 1.0 means the hardware
        achieves less than the stream-benchmark BW for this workload.
        """
        plan_cost = self.estimate_plan(plan, workload)
        report: Dict[int, Dict[str, float]] = {}
        for sc in plan_cost.stage_costs:
            sid = sc.stage_id
            if sid not in measured_ms:
                continue
            m = measured_ms[sid]

            pred_dec   = sc.decode_total_s  * 1000
            pred_pre   = sc.prefill_total_s * 1000
            actual_dec = m.get("decode_ms",  0.0)
            actual_pre = m.get("prefill_ms", 0.0)

            err_dec = (
                abs(pred_dec - actual_dec) / actual_dec * 100
                if actual_dec > 0 else float("nan")
            )
            err_pre = (
                abs(pred_pre - actual_pre) / actual_pre * 100
                if actual_pre > 0 else float("nan")
            )

            # Suggested efficiency: raw_memory_s / actual_s (if memory-bound).
            # sc.decode_memory_s is raw (bytes / bw, no efficiency applied).
            sug_eff_dec = None
            if actual_dec > 0 and sc.decode_memory_s > 0 and sc.decode_regime == "memory":
                sug_eff_dec = round(sc.decode_memory_s / (actual_dec / 1000), 3)

            sug_eff_pre = None
            if actual_pre > 0 and sc.prefill_memory_s > 0:
                sug_eff_pre = round(sc.prefill_memory_s / (actual_pre / 1000), 3)

            report[sid] = {
                "predicted_decode_ms":          round(pred_dec, 2),
                "actual_decode_ms":             round(actual_dec, 2),
                "decode_error_pct":             round(err_dec, 1),
                "predicted_prefill_ms":         round(pred_pre, 2),
                "actual_prefill_ms":            round(actual_pre, 2),
                "prefill_error_pct":            round(err_pre, 1),
                "decode_regime":                sc.decode_regime,
                "suggested_efficiency_memory":  sug_eff_dec,
                "suggested_efficiency_memory_prefill": sug_eff_pre,
            }
        return report

    # ── Internal: memory sizing ───────────────────────────────────────────────

    def _stage_memory(
        self, stage: StageSpec, workload: WorkloadSpec
    ) -> Tuple[int, int, int, int]:
        """
        Returns (param_bytes, act_prefill, act_decode, kv_bytes).

        Activation memory = the hidden-state tensor at the stage boundary:
            B × S × H × dtype_size
        This is tight — intermediate tensors (QKV projections, MLP activations)
        are transient and do not accumulate across layers.

        KV memory = all layers in this stage over the full decode context:
            sum(kv_bytes_per_token) × B × context_len
        """
        B   = workload.batch_size
        S   = workload.prefill_seq_len
        ctx = workload.decode_context_len

        param_bytes = sum(u.param_bytes for u in stage.units)

        if self.model_config is not None:
            H   = self.model_config.hidden_size
            dsz = self.model_config.dtype_size
            act_prefill = B * S * H * dsz
            act_decode  = B * 1 * H * dsz
        else:
            # Fallback: stored activation_bytes has analysis B×S baked in;
            # use as-is (approximate but avoids crashing).
            act_prefill = sum(u.activation_bytes for u in stage.units)
            act_decode  = sum(u.activation_bytes for u in stage.units)

        kv_bytes = sum(u.kv_bytes_per_token for u in stage.units) * B * ctx

        return param_bytes, act_prefill, act_decode, kv_bytes

    # ── Internal: helpers ─────────────────────────────────────────────────────

    def _flops_split(
        self, stage: StageSpec, workload: WorkloadSpec, for_prefill: bool
    ) -> Tuple[float, float]:
        """
        Returns (proj_flops, attn_flops) for the stage.

        Projections (Q/K/V/O + gate/up/down) scale O(S) for prefill, O(1) for
        decode.  Attention (QK^T dot products) scales O(S²) for prefill and
        O(context_len) for decode.  Keeping them separate lets each get its own
        roofline — projection GEMMs can flip compute-bound on GPU at large S
        while attention always remains memory-BW-bound (low arithmetic intensity).
        """
        if self.model_config is None:
            raw = float(sum(
                u.flops_prefill if for_prefill else u.flops_decode
                for u in stage.units
            ))
            # Rough split when ModelConfig is unavailable:
            # ~80% projection, ~20% attention (typical at short sequence).
            return raw * 0.8, raw * 0.2

        mc  = self.model_config
        B   = workload.batch_size
        H   = mc.hidden_size
        I   = mc.intermediate_size
        kv  = mc.num_kv_heads * mc.head_dim

        proj_total = 0.0
        attn_total = 0.0
        for u in stage.units:
            if u.unit_type == "transformer_block":
                if for_prefill:
                    S = workload.prefill_seq_len
                    proj_total += 2 * B * S * (H*H + H*kv + H*kv + H*H + H*I + H*I + I*H)
                    # Causal self-attention: O(S²) per head
                    attn_total += 2 * B * mc.num_attention_heads * S * S * mc.head_dim
                else:
                    ctx = workload.decode_context_len
                    proj_total += 2 * B * (H*H + H*kv + H*kv + H*H + H*I + H*I + I*H)
                    # One query token attends over ctx cached tokens
                    attn_total += 4 * B * mc.num_attention_heads * ctx * mc.head_dim
            else:
                # Embedding, norm, lm_head — not attention
                proj_total += float(
                    u.flops_prefill if for_prefill else u.flops_decode
                )
        return proj_total, attn_total

    def _effective_kv_bw(self, device: DeviceProfile, kv_bytes: int) -> float:
        """
        Effective bandwidth for KV cache reads on CPU, accounting for L3 cache.

        For short contexts the KV working set fits in L3; effective BW is the
        L3 bandwidth (typically 3–6× DRAM).  As context grows beyond l3_size,
        the fraction served from DRAM increases linearly.

        GPU devices and CPU devices without l3_bandwidth set use mem_bandwidth.
        """
        if (device.device_type != "cpu"
                or device.l3_bandwidth <= 0
                or device.l3_size_bytes <= 0
                or kv_bytes <= 0):
            return device.mem_bandwidth

        if kv_bytes <= device.l3_size_bytes:
            return device.l3_bandwidth
        # Fraction that fits in L3 vs DRAM (simplified linear blend)
        l3_frac = device.l3_size_bytes / kv_bytes
        return l3_frac * device.l3_bandwidth + (1.0 - l3_frac) * device.mem_bandwidth

    # ── Internal: time estimates ──────────────────────────────────────────────

    def _decode_time(
        self,
        stage: StageSpec,
        workload: WorkloadSpec,
        device: DeviceProfile,
    ) -> Tuple[float, float, float]:
        """
        Returns (compute_s_raw, memory_s_raw, base_s) for one decode step.

        Two separate rooflines are applied and summed:
          1. Projection roofline — bottleneck: weight bytes from DRAM
          2. Attention roofline  — bottleneck: KV bytes (DRAM or L3 on CPU)

        compute_s_raw and memory_s_raw are the aggregate un-efficiencied values
        stored in StageCost for introspection and validate() calibration.
        base_s applies efficiency_compute / efficiency_memory.
        """
        B   = workload.batch_size
        ctx = workload.decode_context_len

        param_bytes  = sum(u.param_bytes for u in stage.units)
        kv_per_layer = sum(u.kv_bytes_per_token for u in stage.units)
        kv_read      = kv_per_layer * B * ctx
        kv_write     = kv_per_layer * B * 1
        kv_bytes     = kv_read + kv_write

        proj_flops, attn_flops = self._flops_split(stage, workload, for_prefill=False)

        eff_c    = device.efficiency_compute
        eff_m    = device.efficiency_memory
        peak     = device.peak_flops_decode
        bw       = device.mem_bandwidth
        eff_kv_bw = self._effective_kv_bw(device, kv_bytes)

        # ── Projection roofline ───────────────────────────────────────────────
        proj_compute_s = proj_flops / (peak * eff_c) if peak > 0 else 0.0
        proj_memory_s  = param_bytes / (bw  * eff_m) if bw  > 0 else 0.0
        proj_s = max(proj_compute_s, proj_memory_s)

        # ── Attention roofline ────────────────────────────────────────────────
        attn_compute_s = attn_flops / (peak    * eff_c) if peak     > 0 else 0.0
        attn_memory_s  = kv_bytes   / (eff_kv_bw * eff_m) if eff_kv_bw > 0 else 0.0
        attn_s = max(attn_compute_s, attn_memory_s)

        base_s = proj_s + attn_s

        # Raw aggregate values for StageCost introspection (no efficiency)
        compute_s_raw = (proj_flops + attn_flops) / peak if peak > 0 else 0.0
        memory_s_raw  = (param_bytes + kv_bytes)  / bw  if bw  > 0 else 0.0

        return compute_s_raw, memory_s_raw, base_s

    def _prefill_time(
        self,
        stage: StageSpec,
        workload: WorkloadSpec,
        device: DeviceProfile,
    ) -> Tuple[float, float, float]:
        """
        Returns (compute_s_raw, memory_s_raw, base_s) for a prefill pass.

        Two separate rooflines:
          1. Projection roofline — weight bytes + input hidden state (CPU only)
          2. Attention roofline  — KV write (+ activation traffic for CPU)

        GPU attention uses flash-attention style fused kernels: Q/K/V tiles stay
        in SRAM, so there is no DRAM traffic for intermediate activations.
        CPU attention does read/write the hidden state between layers.
        """
        B = workload.batch_size
        S = workload.prefill_seq_len

        param_bytes = sum(u.param_bytes for u in stage.units)
        kv_write    = sum(u.kv_bytes_per_token for u in stage.units) * B * S

        if self.model_config is not None:
            H   = self.model_config.hidden_size
            dsz = self.model_config.dtype_size
            if device.device_type == "cuda":
                # Flash attention: activations stay in SRAM, no DRAM round-trips
                act_traffic = 0
            else:
                n_blocks = sum(1 for u in stage.units if u.unit_type == "transformer_block")
                # read input + write output of the hidden state per layer
                act_traffic = 2 * B * S * H * dsz * max(n_blocks, 1)
        else:
            if device.device_type == "cuda":
                act_traffic = 0
            else:
                act_traffic = sum(u.activation_bytes for u in stage.units) * 2

        proj_flops, attn_flops = self._flops_split(stage, workload, for_prefill=True)

        eff_c = device.efficiency_compute
        eff_m = device.efficiency_memory
        peak  = device.peak_flops_prefill
        bw    = device.mem_bandwidth

        # ── Projection roofline ───────────────────────────────────────────────
        proj_compute_s = proj_flops / (peak * eff_c) if peak > 0 else 0.0
        proj_memory_s  = param_bytes / (bw   * eff_m) if bw   > 0 else 0.0
        proj_s = max(proj_compute_s, proj_memory_s)

        # ── Attention roofline ────────────────────────────────────────────────
        attn_mem_bytes = kv_write + act_traffic
        attn_compute_s = attn_flops    / (peak * eff_c) if peak > 0 else 0.0
        attn_memory_s  = attn_mem_bytes / (bw   * eff_m) if bw   > 0 else 0.0
        attn_s = max(attn_compute_s, attn_memory_s)

        base_s = proj_s + attn_s

        # Raw aggregate values for StageCost introspection (no efficiency)
        total_bytes   = param_bytes + act_traffic + kv_write
        compute_s_raw = (proj_flops + attn_flops) / peak if peak > 0 else 0.0
        memory_s_raw  = total_bytes                / bw  if bw  > 0 else 0.0

        return compute_s_raw, memory_s_raw, base_s

    def _boundary_comm(
        self,
        stage: StageSpec,
        next_stage: Optional[StageSpec],
        workload: WorkloadSpec,
    ) -> Tuple[float, float]:
        """
        PCIe transfer time for the activation tensor at the stage boundary.

        The only tensor crossing devices is the hidden state:
          prefill: B × S × H × dtype_size
          decode:  B × 1 × H × dtype_size
        """
        if next_stage is None or stage.device_id == next_stage.device_id:
            return 0.0, 0.0

        link_key = (stage.device_id, next_stage.device_id)
        if link_key in self.links:
            link    = self.links[link_key]
            bw      = link.bandwidth_bytes_per_s
            latency = link.latency_s
        else:
            bw      = 1e9     # 1 GB/s conservative fallback
            latency = 1e-3    # 1 ms fallback

        if self.model_config is not None:
            H   = self.model_config.hidden_size
            dsz = self.model_config.dtype_size
        else:
            H   = 4096
            dsz = workload.dtype_size

        B = workload.batch_size
        S = workload.prefill_seq_len

        pre_bytes = B * S * H * dsz
        dec_bytes = B * 1 * H * dsz

        return (latency + pre_bytes / bw, latency + dec_bytes / bw)
