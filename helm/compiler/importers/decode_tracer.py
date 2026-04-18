import inspect
import torch
import torch.fx as fx
import torch.nn as nn

from transformers.cache_utils import DynamicCache


class DecodeTracer:
    """
    Traces a single decode step with KV cache support.

    Decoder layers are FX *leaf modules* so their real forward() (including
    DynamicCache.update()) runs natively at each step.  The DynamicCache is
    kept entirely off the FX graph — it is injected before each layer call
    via a registered forward_pre_hook, avoiding FX code-gen trying to
    repr() the cache object.

    Graph contract:
        inputs:  (input_ids, attention_mask, position_ids, cache_position)
        outputs: logits tensor
    """

    class _HelmTracer(fx.Tracer):
        def is_leaf_module(self, module, module_qualified_name):
            if "embed_tokens" in module_qualified_name:
                return True
            if "rotary_emb" in module_qualified_name:
                return True
            if "lm_head" in module_qualified_name:
                return True
            # Decoder layers are leaves so DynamicCache.update() runs
            # natively at runtime (list.append is invisible to FX).
            if module.__class__.__name__.endswith("DecoderLayer"):
                return True
            return super().is_leaf_module(module, module_qualified_name)

    class _Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.past_key_values = DynamicCache()

            first_layer = model.model.layers[0]
            layer_sig = inspect.signature(first_layer.forward)
            self._kv_kwarg = (
                "past_key_value"
                if "past_key_value" in layer_sig.parameters
                else "past_key_values"
            )
            self._use_pos_embed = "position_embeddings" in layer_sig.parameters

            # Pre-hooks inject the DynamicCache before each layer's forward.
            # Hooks are NOT called during FX tracing (leaf modules are opaque
            # to the tracer), only at real runtime.
            # NOTE: back-reference model._helm_decode_wrapper is set AFTER
            # tracing in DecodeTracer.trace() to avoid FX collect_tensor_attrs
            # entering an infinite cycle through the circular reference.

            self._kv_hooks = []
            kv_kwarg = self._kv_kwarg
            for layer in model.model.layers:
                def _hook(module, args, kwargs, _self=self, _kv=kv_kwarg):
                    kwargs[_kv] = _self.past_key_values
                    kwargs["use_cache"] = True
                    return args, kwargs
                self._kv_hooks.append(
                    layer.register_forward_pre_hook(_hook, with_kwargs=True)
                )

        def forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            cache_position,
        ):
            hidden_states = self.model.model.embed_tokens(input_ids)
            attention_mask = attention_mask.to(hidden_states.dtype)

            position_embeddings = None
            if hasattr(self.model.model, "rotary_emb"):
                position_embeddings = self.model.model.rotary_emb(
                    hidden_states, position_ids=position_ids
                )

            for layer in self.model.model.layers:
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                }
                if self._use_pos_embed and position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings

                # past_key_value is NOT in layer_kwargs here — injected by
                # _hook at runtime so DynamicCache never enters the FX graph.
                # New transformers (≥5.0) returns a plain tensor; older versions
                # returned a tuple (hidden_states, present_key_value).
                # isinstance(Proxy, ...) is False at trace time so no getitem
                # node is emitted; at runtime we unpack only when needed.
                layer_out = layer(hidden_states, **layer_kwargs)
                hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            if hasattr(self.model.model, "norm"):
                hidden_states = self.model.model.norm(hidden_states)

            logits = self.model.lm_head(hidden_states)
            return logits  # cache updated in-place via hooks

    def __init__(self, model):
        self.model = model

    def trace(self, example_inputs):
        """
        example_inputs:
            (input_ids, attention_mask, position_ids, cache_position)
        """
        tracer = self._HelmTracer()
        wrapper = self._Wrapper(self.model)
        graph = tracer.trace(wrapper)

        # Set back-reference AFTER tracing to avoid a circular-reference cycle
        # that would cause FX collect_tensor_attrs to recurse infinitely:
        #   wrapper.model._helm_decode_wrapper.model._helm_decode_wrapper ...
        self.model._helm_decode_wrapper = wrapper

        gm = fx.GraphModule(wrapper, graph)
        gm.graph.lint()
        gm.recompile()

        return gm

    @staticmethod
    def build_dummy_inputs(device="cpu", batch_size=1, dtype=torch.float32):
        """
        Construct decode dummy inputs with s=1.
        past_key_values is handled via the wrapper attribute, not here.
        """
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)

        expanded_mask = attention_mask[:, None, None, :]
        min_val = torch.finfo(dtype).min
        causal_mask = (1.0 - expanded_mask.to(dtype)) * min_val

        position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        cache_position = torch.tensor([0], dtype=torch.long, device=device)

        return input_ids, causal_mask, position_ids, cache_position
