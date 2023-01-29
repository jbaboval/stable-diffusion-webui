import numpy
import torch
#import intel_extension_for_pytorch as ipex
from modules import devices
from modules.accelerators.base_accelerator import BaseAccelerator

class OneApiAccelerator(BaseAccelerator):

    def __init__(self):
        import intel_extension_for_pytorch
        global ipex
        ipex = intel_extension_for_pytorch
        torch.use_deterministic_algorithms = True
        self.device = torch.device("xpu")
        return

    @classmethod
    def discover(cls):
        try:
            import intel_extension_for_pytorch
            if torch.xpu.is_available():
                print("OneAPI is available")
                cls._instance = cls()
                return cls._instance
        except Exception as e:
            pass
        
        return None

    def get_device(self):
        return "xpu"

    def gc(self):
        torch.xpu.empty_cache()
        
    def empty_cache(self):
        torch.xpu.empty_cache()

    def memory_summary(self):
        return torch.xpu.memory_summary()

    def autocast(self, dtype=None):
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)
    
    def optimize(self, model, dtype):
        model.training = False
        return ipex.optimize(model, dtype)

    def reset_peak_memory_stats(self):
        return torch.xpu.reset_peak_memory_stats()

    def enable_tf32(self):
        #ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.TF32)
        ipex.set_fp32_math_mode(device="xpu", mode=ipex.FP32MathMode.BF32)
        return
    
    def get_total_memory(self):
        return 16 * 1024 * 1024 * 1024

    def memory_stats(self):
        return torch.xpu.memory_stats(self.device)
    
    def get_rng_state_all(self):
        return torch.xpu.get_rng_state_all()
    
    def set_rng_state(self, state):
        torch.xpu.set_rng_state_all(state)

    def manual_seed(self, seed):
        torch.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
        numpy.random.seed(seed)
    
    def randn(self, shape, device=torch.device("cpu")):
        return torch.randn(shape, device=device).to(self.device)
    
    def randn_like(self, x, device=None):
        if device is None:
            device = self.device
        return torch.randn_like(x, device=devices.cpu).to(x.device)
    
    @property
    def amp(self):
        return torch.xpu.amp