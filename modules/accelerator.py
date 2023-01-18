import torch

from modules.cuda_accelerator import CudaAccelerator
from modules.one_api_accelerator import OneApiAccelerator

impl = None
    
if torch.cuda.is_available():
    print("CUDA is available")
    impl = CudaAccelerator()
else:
    try:
        import intel_extension_for_pytorch
        if torch.xpu.is_available():
            print("OneAPI is available")
            impl = OneApiAccelerator()
    except Exception as e:
        print(f"Exception: {e}")
        pass

def accelerated():
    return impl is not None

def get_device():
    return impl.get_device()

def autocast(dtype):
    return impl.autocast(dtype)

def optimize(model, dtype):
    return impl.optimize(model, dtype)

def memory_stats(device=None):
    return impl.memory_stats(device)

def memory_summary():
    return impl.memory_summary()

def reset_peak_memory_stats():
    return impl.reset_peak_memory_stats()

def get_free_memory():
    return impl.get_free_memory()

def get_total_memory():
    return impl.get_total_memory()
    
def empty_cache():
    torch.cuda.empty_cache()

def manual_seed(seed):
    return impl.manual_seed(seed)
    
def gc():
    impl.gc()
    
def enable_tf32():
    return impl.enable_tf32()
