import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx as fx

from ..IR.graph import HelmGraph, HelmNode


TensorLike = Union[torch.Tensor, Sequence[Any], Dict[str, Any], Any]


@dataclass
class HybridAnalysisSummary:
    num_nodes: int
    num_nodes_with_shapes: int
    total_activation_bytes: int
    total_param_bytes: int
    total_flops_prefill: int
    total_flops_decode: int
    total_kv_bytes_per_token: int


class HybridAnalyzer:
    """
    Pass: Hybrid Cost Analysis

    Responsibilities:
    - run the FX GraphModule on example inputs to capture per-node outputs
    - annotate node shapes and activation sizes
    - annotate per-module parameter bytes
    - estimate block-level FLOPs for prefill/decode
    - estimate KV-cache growth per token

    Notes:
    - This is intentionally a planner-oriented estimator, not an exact profiler.
    - It is safe for early HELM development because it produces useful cost signals
      even when some node types remain conservatively zero-cost.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        helm_graph: HelmGraph,
        model: torch.nn.Module,
        tokenizer: Optional[Any] = None,
    ):
        self.gm = gm
        self.helm_graph = helm_graph
        self.model = model
        self.tokenizer = tokenizer

        self.config = self._build_model_config()

    # ============================================================
    # Public API
    # ============================================================

    def run(self, example_inputs: Any) -> HybridAnalysisSummary:
        self._reset_annotations()
        node_to_output = self._propagate_shapes(example_inputs)
        self._annotate_node_shapes_and_activations(node_to_output)
        self._annotate_module_costs()
        summary = self._build_summary()
        self._attach_summary_to_graph(summary)
        return summary

    def export_summary(self, path: str) -> None:
        data = []
        for node in self.helm_graph.nodes:
            data.append(
                {
                    "node_id": node.id,
                    "node_name": node.name,
                    "fx_node_name": node.fx_node_name,
                    "op_type": node.op_type,
                    "target": str(node.target),
                    "module_path": node.module_path,
                    "layer_id": node.layer_id,
                    "block_id": node.block_id,
                    "input_shapes": node.input_shapes,
                    "output_shapes": node.output_shapes,
                    "activation_bytes": node.activation_bytes,
                    "param_bytes": node.param_bytes,
                    "flops_prefill": node.flops_prefill,
                    "flops_decode": node.flops_decode,
                    "kv_bytes_per_token": node.kv_bytes_per_token,
                    "output_dtype": str(node.output_dtype) if node.output_dtype is not None else None,
                }
            )

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ============================================================
    # Setup / Reset
    # ============================================================

    def _reset_annotations(self) -> None:
        for node in self.helm_graph.nodes:
            node.input_shapes = []
            node.output_shapes = []
            node.flops = 0
            node.flops_prefill = 0
            node.flops_decode = 0
            node.activation_bytes = 0
            node.param_bytes = 0
            node.bytes_read = 0
            node.bytes_written = 0
            node.kv_bytes_per_token = 0
            node.tp_comm_bytes_prefill = 0
            node.tp_comm_bytes_decode = 0
            node.output_dtype = None

            # helper attrs used by analysis/planning
            node.batch_size = 1
            node.sequence_length = 1

    def _build_model_config(self) -> Dict[str, Any]:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            dtype = self._infer_model_dtype()
            dtype_size = torch.tensor([], dtype=dtype).element_size()
            return {
                "hidden_size": 0,
                "num_hidden_layers": 0,
                "num_attention_heads": 0,
                "num_key_value_heads": 0,
                "intermediate_size": 0,
                "vocab_size": 0,
                "dtype": dtype,
                "dtype_size": dtype_size,
            }

        dtype = getattr(cfg, "torch_dtype", None)
        if dtype is None:
            dtype = self._infer_model_dtype()

        dtype_size = torch.tensor([], dtype=dtype).element_size()

        hidden_size = getattr(cfg, "hidden_size", 0)
        intermediate_size = getattr(cfg, "intermediate_size", 0)
        num_heads = getattr(cfg, "num_attention_heads", 0)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        vocab_size = getattr(cfg, "vocab_size", 0)
        num_hidden_layers = getattr(cfg, "num_hidden_layers", 0)

        return {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "dtype": dtype,
            "dtype_size": dtype_size,
        }

    def _infer_model_dtype(self) -> torch.dtype:
        try:
            first_param = next(self.model.parameters())
            return first_param.dtype
        except StopIteration:
            return torch.float16

    # ============================================================
    # Shape Propagation
    # ============================================================

    def _propagate_shapes(self, example_inputs: Any) -> Dict[fx.Node, Any]:
        """
        Executes the GraphModule via an FX interpreter and captures every node output.
        """

        node_to_output: Dict[fx.Node, Any] = {}

        class ShapeInterpreter(fx.Interpreter):
            def run_node(self_inner, n: fx.Node):
                result = super().run_node(n)
                node_to_output[n] = result
                return result

        interp = ShapeInterpreter(self.gm)

        with torch.no_grad():
            if isinstance(example_inputs, dict):
                interp.run(**example_inputs)
            elif isinstance(example_inputs, (tuple, list)):
                interp.run(*example_inputs)
            else:
                interp.run(example_inputs)

        return node_to_output

    def _annotate_node_shapes_and_activations(self, node_to_output: Dict[fx.Node, Any]) -> None:
        for helm_node in self.helm_graph.nodes:
            fx_node = helm_node.fx_node
            if fx_node not in node_to_output:
                continue

            out = node_to_output[fx_node]

            helm_node.output_shapes = self._extract_shapes(out)
            helm_node.activation_bytes = self._activation_size_bytes(out)
            helm_node.output_dtype = self._extract_dtype(out)

            helm_node.input_shapes = []
            for input_fx_node in fx_node.all_input_nodes:
                if input_fx_node in node_to_output:
                    inp = node_to_output[input_fx_node]
                    helm_node.input_shapes.extend(self._extract_shapes(inp))

            batch_size, sequence_length = self._infer_batch_and_sequence(out)
            helm_node.batch_size = batch_size
            helm_node.sequence_length = sequence_length

    # ============================================================
    # Cost Annotation
    # ============================================================

    def _annotate_module_costs(self) -> None:
        for helm_node in self.helm_graph.nodes:
            fx_node = helm_node.fx_node

            if fx_node.op == "call_module":
                submodule = self.gm.get_submodule(fx_node.target)
                helm_node.param_bytes = self._module_param_bytes(submodule)

                # Conservative default I/O traffic model
                helm_node.bytes_written = helm_node.activation_bytes
                helm_node.bytes_read = sum(
                    self._numel_from_shape(shape) * self._dtype_size_from_node(helm_node)
                    for shape in helm_node.input_shapes
                    if self._is_plain_shape(shape)
                )

                if self._is_transformer_block(fx_node.target):
                    b = getattr(helm_node, "batch_size", 1)
                    s = max(getattr(helm_node, "sequence_length", 1), 1)

                    stats = self._estimate_block_costs(batch_size=b, seq_len=s)
                    helm_node.flops_prefill = stats["flops_prefill"]
                    helm_node.flops_decode = stats["flops_decode"]
                    helm_node.kv_bytes_per_token = stats["kv_bytes_per_token"]

                elif self._is_lm_head(fx_node.target):
                    b = getattr(helm_node, "batch_size", 1)
                    s = max(getattr(helm_node, "sequence_length", 1), 1)
                    h = self.config["hidden_size"]
                    v = self.config["vocab_size"]

                    if h > 0 and v > 0:
                        helm_node.flops_prefill = int(2 * b * s * h * v)
                        helm_node.flops_decode = int(2 * b * 1 * h * v)

                elif self._is_embedding_module(fx_node.target):
                    # Embeddings are memory-heavy but relatively light on FLOPs.
                    # Keep FLOPs at zero for v1 and let param/activation bytes drive planning.
                    pass

            else:
                # non-module nodes still carry activation metadata
                helm_node.bytes_written = helm_node.activation_bytes
                helm_node.bytes_read = sum(
                    self._numel_from_shape(shape) * self._dtype_size_from_node(helm_node)
                    for shape in helm_node.input_shapes
                    if self._is_plain_shape(shape)
                )

    def _estimate_block_costs(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """
        Coarse transformer-block estimator.

        Assumptions:
        - decoder-only transformer block
        - fused attention treated analytically
        - decode uses one query token attending over `seq_len` context
        """

        b = max(batch_size, 1)
        s = max(seq_len, 1)
        h = self.config["hidden_size"]
        i = self.config["intermediate_size"]

        if h <= 0 or i <= 0:
            return {
                "flops_prefill": 0,
                "flops_decode": 0,
                "kv_bytes_per_token": 0,
            }

        # Prefill:
        # q/k/v/o projections ~ 4 * B * S * H * H
        attn_linear = 4 * b * s * h * h

        # attention scores + weighted value accumulation
        # coarse dense attention estimate
        attn_scores = 2 * b * s * s * h

        # MLP: gate/up/down style coarse estimate
        mlp = 3 * b * s * h * i

        flops_prefill = int(attn_linear + attn_scores + mlp)

        # Decode:
        # one new token attends over s context
        attn_linear_decode = 4 * b * 1 * h * h
        attn_scores_decode = 2 * b * 1 * s * h
        mlp_decode = 3 * b * 1 * h * i

        flops_decode = int(attn_linear_decode + attn_scores_decode + mlp_decode)

        kv_heads = max(self.config["num_key_value_heads"], 1)
        attn_heads = max(self.config["num_attention_heads"], 1)
        head_dim = h // attn_heads if attn_heads > 0 else 0
        dtype_size = self.config["dtype_size"]

        kv_bytes_per_token = int(2 * kv_heads * head_dim * dtype_size)

        return {
            "flops_prefill": flops_prefill,
            "flops_decode": flops_decode,
            "kv_bytes_per_token": kv_bytes_per_token,
        }

    # ============================================================
    # Summary
    # ============================================================

    def _build_summary(self) -> HybridAnalysisSummary:
        return HybridAnalysisSummary(
            num_nodes=len(self.helm_graph.nodes),
            num_nodes_with_shapes=sum(1 for n in self.helm_graph.nodes if n.output_shapes),
            total_activation_bytes=sum(n.activation_bytes for n in self.helm_graph.nodes),
            total_param_bytes=sum(n.param_bytes for n in self.helm_graph.nodes),
            total_flops_prefill=sum(n.flops_prefill for n in self.helm_graph.nodes),
            total_flops_decode=sum(n.flops_decode for n in self.helm_graph.nodes),
            total_kv_bytes_per_token=sum(n.kv_bytes_per_token for n in self.helm_graph.nodes),
        )

    def _attach_summary_to_graph(self, summary: HybridAnalysisSummary) -> None:
        self.helm_graph.hardware_meta.setdefault("analysis", {})
        self.helm_graph.hardware_meta["analysis"]["hybrid"] = {
            "num_nodes": summary.num_nodes,
            "num_nodes_with_shapes": summary.num_nodes_with_shapes,
            "total_activation_bytes": summary.total_activation_bytes,
            "total_param_bytes": summary.total_param_bytes,
            "total_flops_prefill": summary.total_flops_prefill,
            "total_flops_decode": summary.total_flops_decode,
            "total_kv_bytes_per_token": summary.total_kv_bytes_per_token,
        }

    # ============================================================
    # Helpers
    # ============================================================

    def _module_param_bytes(self, module: torch.nn.Module) -> int:
        return sum(p.numel() * p.element_size() for p in module.parameters())

    def _is_transformer_block(self, target: str) -> bool:
        return (
            ".layers." in target
            or ".blocks." in target
            or ".h." in target
        )

    def _is_lm_head(self, target: str) -> bool:
        return "lm_head" in target or "output_layer" in target

    def _is_embedding_module(self, target: str) -> bool:
        return (
            "embed_tokens" in target
            or "wte" in target
            or "embeddings" in target
        )

    def _activation_size_bytes(self, obj: TensorLike) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(self._activation_size_bytes(x) for x in obj)
        if isinstance(obj, dict):
            return sum(self._activation_size_bytes(v) for v in obj.values())
        return 0

    def _extract_shapes(self, obj: TensorLike) -> List[List[int]]:
        """
        Returns a flat list of tensor shapes found in the object.
        """
        shapes: List[List[int]] = []

        def visit(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                shapes.append(list(x.shape))
            elif isinstance(x, (list, tuple)):
                for item in x:
                    visit(item)
            elif isinstance(x, dict):
                for v in x.values():
                    visit(v)

        visit(obj)
        return shapes

    def _extract_dtype(self, obj: TensorLike) -> Optional[torch.dtype]:
        if isinstance(obj, torch.Tensor):
            return obj.dtype
        if isinstance(obj, (list, tuple)):
            for item in obj:
                dt = self._extract_dtype(item)
                if dt is not None:
                    return dt
        if isinstance(obj, dict):
            for v in obj.values():
                dt = self._extract_dtype(v)
                if dt is not None:
                    return dt
        return None

    def _infer_batch_and_sequence(self, obj: TensorLike) -> Tuple[int, int]:
        shapes = self._extract_shapes(obj)
        if not shapes:
            return 1, 1

        # Prefer rank >= 3 tensors (B, S, H)
        for shape in shapes:
            if len(shape) >= 3:
                return int(shape[0]), int(shape[1])

        # Then rank 2 tensors (B, H) -> decode-like single token
        for shape in shapes:
            if len(shape) == 2:
                return int(shape[0]), 1

        return 1, 1

    def _numel_from_shape(self, shape: List[int]) -> int:
        n = 1
        for dim in shape:
            n *= int(dim)
        return n

    def _dtype_size_from_node(self, node: HelmNode) -> int:
        if node.output_dtype is not None:
            return torch.tensor([], dtype=node.output_dtype).element_size()
        return self.config["dtype_size"]

    def _is_plain_shape(self, shape: Any) -> bool:
        return isinstance(shape, list) and all(isinstance(x, int) for x in shape)