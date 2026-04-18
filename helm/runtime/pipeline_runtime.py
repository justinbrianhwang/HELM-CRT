import torch


class PipelineRuntime:
    """
    Sequential CPU-GPU pipeline runtime.

    Architecture:
        - Prefill:  run the decode executor on the full prompt sequence.
                    The DynamicCache embedded in the decode wrapper is
                    populated in-place with KV for all prompt tokens.
        - Decode:   run the decode executor one token at a time.
                    Each call extends the same DynamicCache, giving the
                    model full context over all previous tokens.

    Both GPU stages and CPU stages share the same DynamicCache object
    (it lives as a constant inside the traced wrapper module).  GPU layers
    write their KV to GPU tensors; CPU layers write theirs to CPU tensors.
    No cross-device copies are needed because each layer only accesses its
    own slice of the cache.
    """

    def __init__(
        self,
        prefill_executor,
        decode_executor,
        tokenizer=None,
        dtype=torch.bfloat16,
        kv_offload_mgr=None,
        decode_wrapper=None,
    ):
        self.prefill_executor = prefill_executor   # kept for future use
        self.decode_executor = decode_executor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.kv_offload_mgr = kv_offload_mgr      # KVOffloadManager or None
        self.decode_wrapper = decode_wrapper       # direct ref to _Wrapper for reliable reset

    # ------------------------------------------------------------------ #
    #  Mask helpers
    # ------------------------------------------------------------------ #

    def _build_causal_mask(self, seq_len, device):
        """Full causal mask for prefill: (1, 1, S, S)."""
        min_val = torch.finfo(self.dtype).min
        mask = torch.full((seq_len, seq_len), min_val, device=device, dtype=self.dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]

    def _build_decode_mask(self, total_len, device):
        """Decode mask: (1, 1, 1, total_len).  All prior positions visible."""
        return torch.zeros((1, 1, 1, total_len), device=device, dtype=self.dtype)

    # ------------------------------------------------------------------ #
    #  Prefill
    # ------------------------------------------------------------------ #

    def prefill(self, input_ids):
        """
        Run prefill using the decode executor on the full prompt.

        The decode wrapper's DynamicCache is empty on the first call and
        gets populated in-place with KV for every prompt token.  Subsequent
        decode steps see this full context automatically.
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        causal_mask = self._build_causal_mask(seq_len, device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, dtype=torch.long, device=device)

        outputs = self.decode_executor.run({
            "input_ids": input_ids,
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        })

        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs

    # ------------------------------------------------------------------ #
    #  Decode step
    # ------------------------------------------------------------------ #

    def decode_step(self, input_ids, step_position):
        """
        Single autoregressive decode step.

        The DynamicCache is updated in-place by the stage modules, so
        the model attends to all prompt tokens plus all previously
        generated tokens without any extra bookkeeping here.
        """
        device = input_ids.device

        position_ids = torch.tensor([[step_position]], dtype=torch.long, device=device)
        cache_position = torch.tensor([step_position], dtype=torch.long, device=device)
        decode_mask = self._build_decode_mask(step_position + 1, device)

        outputs = self.decode_executor.run({
            "input_ids": input_ids,
            "attention_mask": decode_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        })

        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs

    # ------------------------------------------------------------------ #
    #  Generation loop
    # ------------------------------------------------------------------ #

    def _reset_decode_cache(self):
        """
        Reset the DynamicCache in the decode wrapper before each generation.

        Fast path: if self.decode_wrapper is set (direct reference to the
        _Wrapper from DecodeTracer), reset it immediately.

        Slow path: search forward_pre_hook __defaults__ across all stage
        submodules to find the wrapper — used when no direct ref is stored.
        """
        from transformers.cache_utils import DynamicCache
        if self.decode_wrapper is not None:
            self.decode_wrapper.past_key_values = DynamicCache()
            return
        for stage in self.decode_executor.stages:
            for _, module in stage.module.named_modules():
                for hook_fn in module._forward_pre_hooks.values():
                    defaults = getattr(hook_fn, "__defaults__", None) or ()
                    for obj in defaults:
                        if hasattr(obj, "past_key_values") and hasattr(obj, "_kv_hooks"):
                            obj.past_key_values = DynamicCache()
                            return

    def generate(self, input_ids, max_new_tokens=8):
        """
        Autoregressive generation.

        1. Prefill  → seeds DynamicCache with all prompt KV, returns logits.
        2. Decode loop → each step extends DynamicCache by one token.
        """
        if self.kv_offload_mgr is not None:
            self.kv_offload_mgr.reset()
        # Always reset DynamicCache regardless of kv_offload_mgr.
        # kv_offload_mgr.reset() only clears the paged KV cache; the
        # DynamicCache embedded in the executor stages must also be cleared so
        # that _update_causal_mask sees past_seen_tokens=0 on the next run.
        self._reset_decode_cache()
        seq_len = input_ids.shape[1]

        logits = self.prefill(input_ids)

        generated = []
        for step in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            step_position = seq_len + step
            # When KV offload is active, the paged KV cache owns all KV state.
            # Reset DynamicCache each step so it never accumulates stale GPU KV
            # tensors that bypass the watermark-eviction logic.
            if self.kv_offload_mgr is not None:
                self._reset_decode_cache()
            logits = self.decode_step(next_token, step_position)

        generations = torch.cat(generated, dim=1)

        if self.tokenizer:
            decoded = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
            print(f"\n[Generated Token Sequence]: {decoded[0]}")

        return generations
