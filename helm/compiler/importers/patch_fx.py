import sys
import torch.fx._symbolic_trace

def apply_fx_patch():
    """
    Monkey patches torch.fx._symbolic_trace._patch_function to bypass the 
    ValueError: code: co_varnames is too small 
    that happens on Python 3.10+ when tracing huge HuggingFace models.
    """
    original_patch_function = torch.fx._symbolic_trace._patch_function

    def patched_patch_function(fn, nargs):
        # The issue occurs because _patch_function attempts to increase co_argcount 
        # but fails to adjust co_varnames correctly on newer Python versions
        # when the number of variables exceeds the length of the tuple.
        # If the number of arguments matches, or if we can just return the original function,
        # we bypass the recreation of the CodeType object.
        co = fn.__code__
        if co.co_argcount >= nargs:
             return fn
             
        # Fallback to original if we really need to patch it
        try:
             return original_patch_function(fn, nargs)
        except ValueError as e:
             if "varnames is too small" in str(e):
                  # This usually happens on HF models with kwargs.
                  # Since symbolic tracing largely ignores kwargs unpacking, returning the original
                  # function is safer than crashing the tracer.
                  return fn
             raise e

    torch.fx._symbolic_trace._patch_function = patched_patch_function
