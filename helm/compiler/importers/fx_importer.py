import re
import torch
from typing import Optional

from ..IR.graph import HelmGraph, HelmNode


class FXImporter:
    _LAYER_PATTERNS = [
        re.compile(r"\blayers\.(\d+)\b"),
        re.compile(r"\bh\.(\d+)\b"),
        re.compile(r"\bblocks\.(\d+)\b"),
    ]

    def __init__(self, gm: torch.fx.GraphModule, helm_graph: HelmGraph):
        self.gm = gm
        self.graph = helm_graph
        self.modules = dict(gm.named_modules())

    def run(self):
        self._assign_module_paths()
        self._assign_layer_and_block_ids()
        self._assign_semantic_tags()
        self._populate_graph_indexes()

    def _assign_module_paths(self):
        for node in self.graph.nodes:
            fx_node = node.fx_node
            if fx_node.op == "call_module":
                node.module_path = str(fx_node.target)
            else:
                node.module_path = self._infer_module_path_from_neighbors(node)

    def _infer_module_path_from_neighbors(self, node: HelmNode) -> str:
        dep_paths = [dep.module_path for dep in node.dependencies if dep.module_path]
        if dep_paths:
            return self._longest_common_module_prefix(dep_paths)
        return ""

    def _longest_common_module_prefix(self, paths):
        split_paths = [p.split(".") for p in paths if p]
        if not split_paths:
            return ""

        prefix = split_paths[0]
        for parts in split_paths[1:]:
            i = 0
            while i < min(len(prefix), len(parts)) and prefix[i] == parts[i]:
                i += 1
            prefix = prefix[:i]
            if not prefix:
                break

        return ".".join(prefix)

    def _extract_layer_id_from_path(self, module_path: str) -> Optional[int]:
        if not module_path:
            return None
        for pattern in self._LAYER_PATTERNS:
            m = pattern.search(module_path)
            if m:
                return int(m.group(1))
        return None

    def _assign_layer_and_block_ids(self):
        for node in self.graph.nodes:
            layer_id = self._extract_layer_id_from_path(node.module_path)
            node.layer_id = layer_id
            node.block_id = layer_id

    def _text_of_node(self, node: HelmNode) -> str:
        return " ".join([
            str(node.name or ""),
            str(node.target or ""),
            str(node.module_path or ""),
            str(node.fx_node_name or ""),
        ]).lower()

    def _assign_semantic_tags(self):
        for node in self.graph.nodes:
            text = self._text_of_node(node)

            if any(k in text for k in ["attn", "attention", "self_attn"]):
                node.is_attention = True

            if any(k in text for k in ["mlp", "ffn", "feed_forward", "down_proj", "up_proj", "gate_proj"]):
                node.is_mlp = True

            if any(k in text for k in ["norm", "layernorm", "rmsnorm"]):
                node.is_norm = True

            if any(k in text for k in ["embed", "embedding", "embed_tokens", "wte"]):
                node.is_embedding = True

            if any(k in text for k in ["lm_head", "output_head", "logits"]):
                node.is_output_head = True

    def _populate_graph_indexes(self):
        self.graph.layer_to_node_ids.clear()
        self.graph.block_to_node_ids.clear()

        for node in self.graph.nodes:
            if node.layer_id is not None:
                self.graph.layer_to_node_ids.setdefault(node.layer_id, []).append(node.id)
            if node.block_id is not None:
                self.graph.block_to_node_ids.setdefault(node.block_id, []).append(node.id)

    def print_summary(self):
        print("\n[FXImporter] Annotation Summary")
        for node in self.graph.nodes:
            print(
                f"{node.name}: module={node.module_path}, "
                f"layer={node.layer_id}, block={node.block_id}, "
                f"attn={node.is_attention}, mlp={node.is_mlp}, norm={node.is_norm}"
            )