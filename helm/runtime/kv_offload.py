"""
KV cache offloading integration for HELM.

Patches model attention modules to use a paged KV cache (KVCacheManager)
with CPU offloading. Streaming attention is used for decode steps so that
older KV pages evicted to CPU are fetched page-by-page without materialising
the full context on GPU.

Prefill  (q_len > 1): SDPA with causal mask  + append_prefill → paged cache
Decode   (q_len == 1): append_decode → paged cache, perform_streaming_attention

The decoder layers remain FX leaf modules — no graph changes needed.
Patches are applied at the class level so every instance (including those
inside stage GraphModules) uses the offloaded path.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from helm.runtime.kv_allocator import KVAllocator
from helm.runtime.kv_cache import KVCacheManager, perform_streaming_attention


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KVOffloadConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int = 64
    dtype: torch.dtype = torch.float16
    # Total GPU KV bytes to keep hot before evicting older pages to CPU.
    # Default: keep ~512 tokens per layer on GPU.
    gpu_watermark_bytes: Optional[int] = None
    # Pre-allocated contiguous KV buffer capacity (tokens).  When the full
    # sequence fits within this budget the fast path uses in-place writes +
    # a single SDPA call — no torch.cat per decode step (same as Accelerate).
    # Set to 0 to disable; defaults to the model's max_position_embeddings.
    cont_capacity: int = 32768

    def __post_init__(self):
        if self.gpu_watermark_bytes is None:
            elem = torch.tensor([], dtype=self.dtype).element_size()
            # 2 tensors (K+V) × kv_heads × head_dim × elem_size × 512 tokens
            per_token_bytes = 2 * self.num_kv_heads * self.head_dim * elem
            self.gpu_watermark_bytes = 512 * per_token_bytes * self.num_layers

    @staticmethod
    def from_model(model, page_size: int = 64,
                   gpu_watermark_bytes: Optional[int] = None,
                   cont_capacity: Optional[int] = None) -> "KVOffloadConfig":
        cfg = model.config
        num_layers   = cfg.num_hidden_layers
        num_kv_heads = getattr(cfg, "num_key_value_heads",
                               cfg.num_attention_heads)
        head_dim     = getattr(cfg, "head_dim",
                               cfg.hidden_size // cfg.num_attention_heads)
        dtype = next(model.parameters()).dtype
        if cont_capacity is None:
            cont_capacity = getattr(cfg, "max_position_embeddings", 32768)
        return KVOffloadConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            gpu_watermark_bytes=gpu_watermark_bytes,
            cont_capacity=cont_capacity,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shared decode step helper
# ─────────────────────────────────────────────────────────────────────────────

def _decode_batched(kvcms, q, k, v, layer_idx: int, scale: float):
    """
    Decode step shared by all architecture forward patches.

    Three execution paths, selected in order:

    1. Contiguous path (all-GPU, sequence within pre-allocated capacity):
       Writes K/V in-place into a pre-allocated [bsz, heads, capacity, head_dim]
       buffer and passes a zero-copy view to SDPA.  No append_decode, no
       repeat_interleave — equivalent to Accelerate's decode path.

    2. Paged GPU path (all-GPU, capacity exceeded or contiguous disabled):
       Assembles batched K/V with torch.cat across pages and issues a single
       batched SDPA call with enable_gqa=True (no KV head expansion).

    3. Paged streaming path (mixed/CPU residency):
       Per-item perform_streaming_attention with async H2D prefetch and online
       softmax.  This is HELM's core offloading path.

    Returns out: [bsz, num_q_heads, 1, head_dim]
    """
    bsz = q.shape[0]

    # ── Path 1: contiguous in-place fast path ────────────────────────────────
    if kvcms[0].use_contiguous:
        if kvcms[0].cont_seq_len < kvcms[0].cont_capacity:
            # Write new token in-place — no append_decode, no paged overhead.
            for i in range(bsz):
                kvcms[i].write_decode_contiguous(layer_idx, k[i:i+1], v[i:i+1])

            # K/V views: zero-copy slice into pre-allocated buffer.
            if bsz == 1:
                K, V = kvcms[0].get_kv_contiguous(layer_idx)
            else:
                Ks, Vs = zip(*(kvcms[i].get_kv_contiguous(layer_idx) for i in range(bsz)))
                K, V = torch.cat(Ks, dim=0), torch.cat(Vs, dim=0)

            # Advance committed token count after the last transformer layer.
            if layer_idx == kvcms[0].cont_num_layers - 1:
                for kvcm in kvcms:
                    kvcm.advance_contiguous()

            # enable_gqa lets the flash-attention kernel handle GQA natively —
            # no repeat_interleave, no extra tensor allocation.
            return F.scaled_dot_product_attention(q, K, V, scale=scale, enable_gqa=True)
        else:
            # Capacity exceeded — bulk-migrate decode tokens to pages, then fall through.
            for kvcm in kvcms:
                kvcm.migrate_contiguous_to_pages()

    # ── Path 2 & 3: paged path ───────────────────────────────────────────────
    all_gpu = all(kvcms[i].all_gpu_resident(layer_idx) for i in range(bsz))

    all_pages = []
    for i in range(bsz):
        kvcms[i].append_decode(layer_idx, k[i:i+1], v[i:i+1], skip_residency=all_gpu)
        all_pages.append(kvcms[i].iterate_layer_pages(layer_idx))

    # ── Path 2: all GPU-resident — single batched SDPA ───────────────────────
    if all_gpu:
        K_list, V_list, seq_len_ref, fast_ok = [], [], None, True
        for pages in all_pages:
            active = [p for p in pages if p.used_tokens > 0]
            if not active:
                fast_ok = False
                break
            K_i = torch.cat([p.k_tensor[:, :, :p.used_tokens, :] for p in active], dim=2)
            V_i = torch.cat([p.v_tensor[:, :, :p.used_tokens, :] for p in active], dim=2)
            if seq_len_ref is None:
                seq_len_ref = K_i.shape[2]
            elif K_i.shape[2] != seq_len_ref:
                fast_ok = False
                break
            K_list.append(K_i)
            V_list.append(V_i)
        if fast_ok and K_list:
            K = torch.cat(K_list, dim=0)
            V = torch.cat(V_list, dim=0)
            return F.scaled_dot_product_attention(q, K, V, scale=scale, enable_gqa=True)

    # ── Path 3: mixed/CPU residency — per-item streaming attention ────────────
    outs = [
        perform_streaming_attention(q[i:i+1], all_pages[i], scale=scale)
        for i in range(bsz)
    ]
    return torch.cat(outs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-architecture patched forward factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_qwen2_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,      # ignored — kvcm handles storage
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attn_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)
        num_kv = k.shape[1]
        groups  = q.shape[1] // num_kv

        if q_len > 1:
            # Prefill: per-item KV storage; batched SDPA for efficiency.
            for i in range(bsz):
                kvcms[i].append_prefill(attn_self.layer_idx, k[i:i+1], v[i:i+1])
                if kvcms[i].use_contiguous:
                    kvcms[i].prefill_contiguous(attn_self.layer_idx, k[i:i+1], v[i:i+1])
            k_exp = k.repeat_interleave(groups, dim=1)
            v_exp = v.repeat_interleave(groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=attention_mask,
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                scale=scale,
            )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


def _make_qwen3_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_norm(attn_self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = attn_self.k_norm(attn_self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)

        if q_len > 1:
            for i in range(bsz):
                kvcms[i].append_prefill(attn_self.layer_idx, k[i:i+1], v[i:i+1])
                if kvcms[i].use_contiguous:
                    kvcms[i].prefill_contiguous(attn_self.layer_idx, k[i:i+1], v[i:i+1])
            k_exp = repeat_kv(k, attn_self.num_key_value_groups)
            v_exp = repeat_kv(v, attn_self.num_key_value_groups)
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=attention_mask,
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                scale=scale,
            )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


def _make_llama_forward(kvcms):
    """kvcms: list of KVCacheManager, one per batch item."""
    def forward(
        attn_self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, attn_self.head_dim)

        q = attn_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = attn_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = attn_self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(attn_self.head_dim)
        groups = q.shape[1] // k.shape[1]

        if q_len > 1:
            for i in range(bsz):
                kvcms[i].append_prefill(attn_self.layer_idx, k[i:i+1], v[i:i+1])
                if kvcms[i].use_contiguous:
                    kvcms[i].prefill_contiguous(attn_self.layer_idx, k[i:i+1], v[i:i+1])
            k_exp = k.repeat_interleave(groups, dim=1)
            v_exp = v.repeat_interleave(groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=attention_mask,
                dropout_p=attn_self.attention_dropout if attn_self.training else 0.0,
                scale=scale,
            )
        else:
            out = _decode_batched(kvcms, q, k, v, attn_self.layer_idx, scale)

        out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        out = attn_self.o_proj(out)
        return out, None

    return forward


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class KVOffloadManager:
    """
    Installs paged-KV attention patches on the model and manages the cache
    lifecycle across generate() calls.

    Supports batch_size > 1: one KVCacheManager is maintained per batch item,
    all sharing a single KVAllocator pool.  Attention patches loop over the
    batch dimension for KV operations and concatenate results.

    Usage:
        mgr = KVOffloadManager(model, KVOffloadConfig.from_model(model), batch_size=4)
        runtime = PipelineRuntime(..., kv_offload_mgr=mgr)
        runtime.generate(input_ids, max_new_tokens=32)  # input_ids: (4, S)
    """

    def __init__(self, model, config: KVOffloadConfig, batch_size: int = 1):
        self.config = config
        self.batch_size = batch_size
        self.allocator = KVAllocator(
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            page_size=config.page_size,
            dtype=config.dtype,
        )
        # Split the GPU watermark budget across batch items so total GPU KV
        # usage stays within the original budget.
        watermark_per_item = max(config.gpu_watermark_bytes // batch_size, 1)
        self.kvcms = [
            KVCacheManager(self.allocator, gpu_high_watermark_bytes=watermark_per_item)
            for _ in range(batch_size)
        ]
        self._patched: dict[str, tuple] = {}  # arch → (cls, orig_forward)
        self._apply_patches(model)

        # Try to pre-allocate contiguous KV buffers for the all-GPU fast path.
        # Falls back gracefully to paged streaming on OOM.
        if config.cont_capacity > 0:
            device = next(model.parameters()).device
            if device.type == "cuda":
                built = [
                    kvcm.build_contiguous(
                        num_layers=config.num_layers,
                        num_kv_heads=config.num_kv_heads,
                        head_dim=config.head_dim,
                        capacity=config.cont_capacity,
                        device=device,
                        dtype=config.dtype,
                    )
                    for kvcm in self.kvcms
                ]
                if not all(built):  # partial OOM — drop all to keep things consistent
                    for kvcm in self.kvcms:
                        kvcm.drop_contiguous()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all paged caches and re-arm contiguous path for next generate()."""
        for kvcm in self.kvcms:
            kvcm.clear()
            kvcm.reset_contiguous()

    def restore(self):
        """Restore original attention forwards (e.g. after generation)."""
        for arch, (cls, orig) in self._patched.items():
            cls.forward = orig
        self._patched.clear()

    def report(self) -> dict:
        return self.allocator.report_usage()

    # ── Patch dispatch ────────────────────────────────────────────────────────

    def _apply_patches(self, model):
        name = model.__class__.__name__.lower()
        if "qwen3" in name:
            self._patch("qwen3")
        elif "qwen2" in name:
            self._patch("qwen2")
        elif "llama" in name or "mistral" in name:
            self._patch("llama")
        else:
            raise NotImplementedError(
                f"KVOffloadManager: no attention patch for {model.__class__.__name__}"
            )

    def _patch(self, arch: str):
        if arch == "qwen2":
            import transformers.models.qwen2.modeling_qwen2 as m
            cls = m.Qwen2Attention
            new_fwd = _make_qwen2_forward(self.kvcms)
        elif arch == "qwen3":
            import transformers.models.qwen3.modeling_qwen3 as m
            cls = m.Qwen3Attention
            new_fwd = _make_qwen3_forward(self.kvcms)
        elif arch == "llama":
            import transformers.models.llama.modeling_llama as m
            cls = m.LlamaAttention
            new_fwd = _make_llama_forward(self.kvcms)
        else:
            raise ValueError(arch)

        self._patched[arch] = (cls, cls.forward)
        cls.forward = new_fwd
