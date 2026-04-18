import torch
from typing import Any

def move_tensor(tensor: Any, device: str) -> Any:
    """
    Moves a tensor to the specified device if necessary.
    In the future, this layer will intercept async PCIe/NVLink transfers.
    """
    if isinstance(tensor, tuple):
        return tuple(move_tensor(t, device) for t in tensor)
    if isinstance(tensor, list):
        return [move_tensor(t, device) for t in tensor]
    if isinstance(tensor, dict):
        return {k: move_tensor(v, device) for k, v in tensor.items()}
        
    if not isinstance(tensor, torch.Tensor):
        return tensor
        
    current_device = str(tensor.device)
    target_device = device
    
    # Normalize device strings (e.g. 'cuda:0' vs 'cuda')
    if current_device == "cpu" and target_device == "cpu":
        return tensor
        
    if "cuda" in current_device and "cuda" in target_device:
        # Check specific index if provided
        curr_idx = current_device.split(":")[1] if ":" in current_device else "0"
        targ_idx = target_device.split(":")[1] if ":" in target_device else "0"
        if curr_idx == targ_idx:
            return tensor
            
    return tensor.to(device)
