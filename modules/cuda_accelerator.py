import torch
from modules.base_accelerator import BaseAccelerator

class CudaAccelerator(BaseAccelerator):
    def __init__(self):
        free, self.total = torch.cuda.mem_get_info()

    def get_device(self):
        return "cuda"

    def get_available_vram(self):
        stats = torch.cuda.memory_stats(self.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total
    
    def memory_stats(self):
        return torch.cuda.memory_stats(self.device)

    def memory_summary(self):
        return torch.cuda.memory_summary()

    def autocast(self, dtype):
        return torch.autocast("cuda")
    
    def optimize(self, model, dtype):
        return model

    def reset_peak_memory_stats(self):
        return torch.cuda.reset_peak_memory_stats()
    
    def get_free_memory(self):
        return torch.cuda.mem_get_info()[0]
    
    def get_total_memory(self):
        return torch.cuda.mem_get_info()[1]
        
    def empty_cache(self):
        torch.cuda.empty_cache()

    def gc(self):
        self.empty_cache()
        torch.cuda.ipc_collect()
        
    def enable_tf32(self):
        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def get_rng_state_all(self):
        return torch.cuda.get_rng_state_all()
    
    def set_rng_state(self, state):
        torch.cuda.set_rng_state(state)
        
    def manual_seed(self, seed):
        torch.manual_seed(seed)

    @property
    def amp(self):
        return torch.cuda.amp