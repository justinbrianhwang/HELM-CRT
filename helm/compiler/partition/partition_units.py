from dataclasses import dataclass, field
from typing import List, Optional

from ..IR.graph import HelmGraph

@dataclass
class PartitionUnit:
    unit_id: int
    unit_type: str
    
    layer_start: Optional[int]
    layer_end: Optional[int]
    
    node_ids: List[int] = field(default_factory=list)
    
    # aggregated cost properties
    flops_prefill: float = 0
    flops_decode: float = 0
    
    param_bytes: int = 0
    activation_bytes: int = 0
    
    kv_bytes_per_token: int = 0
    
    contains_attention: bool = False
    contains_mlp: bool = False
    contains_norm: bool = False

class PartitionUnitBuilder:
    def __init__(self, helm_graph: HelmGraph):
        self.graph = helm_graph
        self.units = []

    def build(self):
        self._build_embedding_unit()
        self._build_layer_units()
        self._build_output_unit()
        return self.units

    def _build_embedding_unit(self):
        embed_nodes = []
        for node in self.graph.nodes:
            if node.is_embedding:
                embed_nodes.append(node.id)

        if not embed_nodes:
            return

        unit = PartitionUnit(
            unit_id=len(self.units),
            unit_type="embedding",
            layer_start=None,
            layer_end=None,
            node_ids=embed_nodes
        )
        self._aggregate_cost(unit)
        self.units.append(unit)

    def _build_layer_units(self):
        layer_ids = sorted(self.graph.layer_to_node_ids.keys())

        for layer_id in layer_ids:
            node_ids = self.graph.layer_to_node_ids[layer_id]

            unit = PartitionUnit(
                unit_id=len(self.units),
                unit_type="transformer_block",
                layer_start=layer_id,
                layer_end=layer_id,
                node_ids=node_ids
            )

            self._aggregate_cost(unit)
            self.units.append(unit)

    def _build_output_unit(self):
        output_nodes = []
        for node in self.graph.nodes:
            if node.is_output_head:
                output_nodes.append(node.id)

        if not output_nodes:
            return

        unit = PartitionUnit(
            unit_id=len(self.units),
            unit_type="output",
            layer_start=None,
            layer_end=None,
            node_ids=output_nodes
        )

        self._aggregate_cost(unit)
        self.units.append(unit)

    def _aggregate_cost(self, unit):
        for nid in unit.node_ids:
            node = self.graph.helm_id_to_node[nid]

            unit.flops_prefill += node.flops_prefill
            unit.flops_decode += node.flops_decode

            unit.param_bytes += node.param_bytes
            unit.activation_bytes += node.activation_bytes

            unit.kv_bytes_per_token += node.kv_bytes_per_token

            if node.is_attention:
                unit.contains_attention = True

            if node.is_mlp:
                unit.contains_mlp = True

            if node.is_norm:
                unit.contains_norm = True

    def print_units(self):
        print("\n[PartitionUnitBuilder] Units")
        for unit in self.units:
            print(
                f"Unit {unit.unit_id} | "
                f"type={unit.unit_type} | "
                f"layers={unit.layer_start}-{unit.layer_end} | "
                f"nodes={len(unit.node_ids)} | "
                f"flops={unit.flops_prefill}"
            )
