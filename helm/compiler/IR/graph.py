import torch
from typing import List, Dict, Any, Union, Optional
from functools import reduce

class HelmEdge:
    """
    Represents a directed edge between two HelmNodes.
    """
    def __init__(
        self,
        src_id: int,
        dst_id: int,
        tensor_shape: Optional[List[int]] = None,
        tensor_bytes: int = 0,
        is_residual: bool = False,
        crosses_layer_boundary: bool = False
    ):
        self.src_id = src_id
        self.dst_id = dst_id
        self.tensor_shape = tensor_shape or []
        self.tensor_bytes = tensor_bytes
        self.is_residual = is_residual
        self.crosses_layer_boundary = crosses_layer_boundary
        
    def __repr__(self):
        return f"HelmEdge({self.src_id} -> {self.dst_id} | Shape: {self.tensor_shape} | Bytes: {self.tensor_bytes})"

class HelmNode:
    """
    Represents a node in the Helm Graph, mirroring an FX node.
    """
    def __init__(self, id: int, name: str, fx_node: torch.fx.Node):
        self.id = id
        self.name = name
        self.fx_node = fx_node
        self.fx_node_name = fx_node.name
        self.op_type = fx_node.op
        self.target = fx_node.target
        self.module_path: str = ""
        self.layer_id: Optional[int] = None
        self.block_id: Optional[int] = None
        
        self.args: List[Union['HelmNode', Any]] = [] 
        self.kwargs: Dict[str, Any] = {}
        self.users: List['HelmNode'] = []
        
        # Core Properties
        self.input_shapes: List[List[int]] = []
        self.output_shapes: List[List[int]] = []
        self.flops: int = 0
        self.flops_prefill: int = 0
        self.flops_decode: int = 0
        
        self.activation_bytes: int = 0
        self.param_bytes: int = 0
        self.bytes_read: int = 0
        self.bytes_written: int = 0
        self.kv_bytes_per_token: int = 0
        self.tp_comm_bytes_prefill: int = 0
        self.tp_comm_bytes_decode: int = 0
        
        self.dependencies: List['HelmNode'] = []
        
        self.output_dtype: Optional[torch.dtype] = None
        
        # Semantic Tags
        self.is_attention: bool = False
        self.is_mlp: bool = False
        self.is_norm: bool = False
        self.is_embedding: bool = False
        self.is_output_head: bool = False
        
        # Backend / Partitioning
        self.device: str = "cpu" # Default placement
        self.assigned_stage: Optional[int] = None
        self.assigned_device_group: Optional[str] = None

    def mark_attention(self):
        self.is_attention = True

    def mark_mlp(self):
        self.is_mlp = True

    def mark_norm(self):
        self.is_norm = True

    def get_output_bytes_str(self) -> str:
        """Formatted string for output bytes (e.g. '4.00 MB')."""
        b = self.activation_bytes
        if b == 0:
            return ""
        if b < 1024:
            return f"{b} B"
        elif b < 1024**2:
            return f"{b/1024:.2f} KB"
        elif b < 1024**3:
            return f"{b/(1024**2):.2f} MB"
        else:
            return f"{b/(1024**3):.2f} GB"

    def __repr__(self):
        # Helper to format args nicely
        fmt_args = []
        for arg in self.args:
            if isinstance(arg, HelmNode):
                fmt_args.append(arg.name)
            else:
                fmt_args.append(str(arg))
        return f"{self.name} = {self.op_type}({self.target}, args={fmt_args}) | Layer: {self.layer_id} | FLOPs: {self.flops_prefill} | Out: {self.get_output_bytes_str()} | Shapes: {self.output_shapes}"

class HelmGraph:
    """
    A mirrored graph representation of the FX Graph.
    """
    def __init__(self, fx_graph: torch.fx.Graph):
        self.nodes: List[HelmNode] = []
        self.edges: List[HelmEdge] = []
        
        self.fx_to_helm: Dict[torch.fx.Node, HelmNode] = {}
        self.helm_id_to_node: Dict[int, HelmNode] = {}
        self.helm_name_to_node: Dict[str, HelmNode] = {}
        self.helm_id_to_fx_name: Dict[int, str] = {}
        self.fx_name_to_helm_id: Dict[str, int] = {}
        
        self.layer_to_node_ids: Dict[int, List[int]] = {}
        self.block_to_node_ids: Dict[int, List[int]] = {}
        
        self.input_node_ids: List[int] = []
        self.output_node_ids: List[int] = []
        
        self.fx_graph = fx_graph # Keep ref
        
        # Global Metadata
        self.hardware_meta: Dict[str, Any] = {}
        
        self._build_from_fx(fx_graph)

    def _extract_dependencies(self, arg: Any) -> Any:
        found_deps = []
        
        def recursive_map(x):
            if isinstance(x, torch.fx.Node):
                if x in self.fx_to_helm:
                    helm_node = self.fx_to_helm[x]
                    found_deps.append(helm_node)
                    return helm_node
                else:
                    return x 
            elif isinstance(x, (list, tuple)):
                return type(x)(recursive_map(item) for item in x)
            elif isinstance(x, dict):
                return {k: recursive_map(v) for k, v in x.items()}
            else:
                return x

        mapped_arg = recursive_map(arg)
        return mapped_arg, found_deps

    def _build_from_fx(self, fx_graph: torch.fx.Graph):
        idx = 0
        for fx_node in fx_graph.nodes:
            helm_name = f"N{idx}"
            helm_node = HelmNode(id=idx, name=helm_name, fx_node=fx_node)
            
            self.nodes.append(helm_node)
            self.fx_to_helm[fx_node] = helm_node
            
            self.helm_id_to_node[helm_node.id] = helm_node
            self.helm_name_to_node[helm_node.name] = helm_node
            self.helm_id_to_fx_name[helm_node.id] = fx_node.name
            self.fx_name_to_helm_id[fx_node.name] = helm_node.id
            
            if fx_node.op == 'placeholder':
                self.input_node_ids.append(helm_node.id)
            elif fx_node.op == 'output':
                self.output_node_ids.append(helm_node.id)
            
            idx += 1

        for helm_node in self.nodes:
            original_args = helm_node.fx_node.args
            
            new_args = []
            all_deps = []
            
            for arg in original_args:
                mapped_arg, deps = self._extract_dependencies(arg)
                new_args.append(mapped_arg)
                all_deps.extend(deps)
            
            helm_node.args = new_args
            helm_node.dependencies = all_deps 

            for dep in all_deps:
                dep.users.append(helm_node)
                
                # Create default HelmEdge
                # Detailed shape/bytes etc. will be populated by analysis passes
                edge = HelmEdge(
                    src_id=dep.id,
                    dst_id=helm_node.id
                )
                self.edges.append(edge)
    
    def print_graph(self):
        print("\n--- Helm Graph (Mirrored) ---")
        if self.hardware_meta:
            print(f"Hardware Context: {self.hardware_meta}")
        for node in self.nodes:
            print(f"{node} [Depends on: {[d.name for d in node.dependencies]}]")
        print("-----------------------------\n")

    def get_node(self, node_id: int) -> HelmNode:
        return self.helm_id_to_node[node_id]

    def get_nodes_by_layer(self, layer_id: int) -> List[HelmNode]:
        ids = self.layer_to_node_ids.get(layer_id, [])
        return [self.helm_id_to_node[i] for i in ids]

    def get_nodes_by_block(self, block_id: int) -> List[HelmNode]:
        ids = self.block_to_node_ids.get(block_id, [])
        return [self.helm_id_to_node[i] for i in ids]

    def topological_nodes(self) -> List[HelmNode]:
        return self.nodes

    def get_outgoing_edges(self, node_id: int) -> List[HelmEdge]:
        return [e for e in self.edges if e.src_id == node_id]

    def get_incoming_edges(self, node_id: int) -> List[HelmEdge]:
        return [e for e in self.edges if e.dst_id == node_id]

    def summary(self):
        print("HelmGraph Summary")
        print("Nodes:", len(self.nodes))
        print("Edges:", len(self.edges))
        print("Inputs:", self.input_node_ids)
        print("Outputs:", self.output_node_ids)
