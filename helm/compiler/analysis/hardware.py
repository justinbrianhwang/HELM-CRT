import torch
import psutil
import os
from ..graph import HelmGraph

class HardwareAnalyzer:
    """
    Pass: Hardware Detection
    Detects available hardware (CPU, GPU, RAM) and attaches metadata to the HelmGraph.
    """
    def __init__(self, graph: HelmGraph):
        self.graph = graph

    def run(self):
        print("\n[HardwareAnalyzer] Detecting System Resources...")
        
        meta = {}
        
        # 1. CPU Info
        meta['cpu_count_physical'] = psutil.cpu_count(logical=False)
        meta['cpu_count_logical'] = psutil.cpu_count(logical=True)
        
        mem = psutil.virtual_memory()
        meta['system_ram_total_gb'] = mem.total / (1024**3)
        meta['system_ram_available_gb'] = mem.available / (1024**3)
        
        # 2. GPU Info
        if torch.cuda.is_available():
            meta['gpu_available'] = True
            meta['gpu_count'] = torch.cuda.device_count()
            
            gpu_info = []
            for i in range(meta['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'total_memory_gb': props.total_memory / (1024**3),
                    'multi_processor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor
                })
            meta['gpus'] = gpu_info
        else:
            meta['gpu_available'] = False
            
        self.graph.hardware_meta = meta
        
        # Visualize / Print Summary
        print(f"  CPU Cores: {meta['cpu_count_physical']} (Phys) / {meta['cpu_count_logical']} (Log)")
        print(f"  System RAM: {meta['system_ram_available_gb']:.2f} GB / {meta['system_ram_total_gb']:.2f} GB")
        
        if meta['gpu_available']:
            for idx, gpu in enumerate(meta['gpus']):
                print(f"  GPU {idx}: {gpu['name']} ({gpu['total_memory_gb']:.2f} GB VRAM, {gpu['multi_processor_count']} SMs)")
        else:
            print("  GPU: None detected.")
