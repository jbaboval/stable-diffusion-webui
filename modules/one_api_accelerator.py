import numpy
import torch
import intel_extension_for_pytorch as ipex
from modules.base_accelerator import BaseAccelerator

class OneApiAccelerator(BaseAccelerator):

    def __init__(self):
        torch.use_deterministic_algorithms = True
        return

    def get_device(self):
        return "xpu"

    def gc(self):
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
    
    @property
    def amp(self):
        return torch.xpu.amp