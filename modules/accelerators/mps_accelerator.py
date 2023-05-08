import numpy
import torch
from packaging import version
from modules.accelerators.base_accelerator import BaseAccelerator

class MPSAccelerator(BaseAccelerator):

    _instance = None

    def __init__(self):
        self.device = torch.device(MPSAccelerator.device_string)
        return

    @classmethod
    @property
    def device_string(cls):
        return "mps"

    @classmethod
    def _has_mps(cls):
        if not getattr(torch, 'has_mps', False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @classmethod
    def discover(cls):
        if cls._has_mps():
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
            
        return None

    def gc(self):
        torch.gc()
        return
        
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

    def einsum_op_mps_v1(self, q, k, v):
        if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
            return einsum_op_compvis(q, k, v)
        else:
            slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
            if slice_size % 4096 == 0:
                slice_size -= 1
            return einsum_op_slice_1(q, k, v, slice_size)

    def einsum_op_mps_v2(self, q, k, v):
        if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
            return einsum_op_compvis(q, k, v)
        else:
            return einsum_op_slice_0(q, k, v, 1)

    def einsum_op(self, q, k, v):
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            return einsum_op_mps_v1(q, k, v)
        return einsum_op_mps_v2(q, k, v)
        
    @property
    def amp(self):
        return torch.xpu.amp

            
# MPS workaround for https://github.com/pytorch/pytorch/issues/90532
orig_tensor_numpy = torch.Tensor.numpy
def _numpy_fix(self, *args, **kwargs):
    if self.requires_grad:
        self = self.detach()
    return orig_tensor_numpy(self, *args, **kwargs)

# MPS workaround for https://github.com/pytorch/pytorch/issues/89784
orig_cumsum = torch.cumsum
orig_Tensor_cumsum = torch.Tensor.cumsum
def _cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == 'mps':
        output_dtype = kwargs.get('dtype', input.dtype)
        if output_dtype == torch.int64:
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
        elif cumsum_needs_bool_fix and output_dtype == torch.bool or cumsum_needs_int_fix and (output_dtype == torch.int8 or output_dtype == torch.int16):
            return cumsum_func(input.to(torch.int32), *args, **kwargs).to(torch.int64)
    return cumsum_func(input, *args, **kwargs)

# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
orig_tensor_to = torch.Tensor.to
def _tensor_to_fix(self, *args, **kwargs):
    if self.device.type != 'mps' and \
    ((len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == 'mps') or \
    (isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps')):
        self = self.contiguous()
    return orig_tensor_to(self, *args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/80800
orig_layer_norm = torch.nn.functional.layer_norm
def _layer_norm_fix(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps':
        args = list(args)
        args[0] = args[0].contiguous()
    return orig_layer_norm(*args, **kwargs)

if MPSAccelerator._has_mps():
    if version.parse(torch.__version__) < version.parse("1.13"):
        # PyTorch 1.13 doesn't need these fixes but unfortunately is slower and has regressions that prevent training from working
        torch.Tensor.to = _tensor_to_fix
        torch.nn.functional.layer_norm = _layer_norm_fix
        torch.Tensor.numpy = _numpy_fix
    elif version.parse(torch.__version__) > version.parse("1.13.1"):
        cumsum_needs_int_fix = not torch.Tensor([1,2]).to(torch.device("mps")).equal(torch.ShortTensor([1,1]).to(torch.device("mps")).cumsum(0))
        cumsum_needs_bool_fix = not torch.BoolTensor([True,True]).to(device=torch.device("mps"), dtype=torch.int64).equal(torch.BoolTensor([True,False]).to(torch.device("mps")).cumsum(0))
        torch.cumsum = lambda input, *args, **kwargs: ( _cumsum_fix(input, orig_cumsum, *args, **kwargs) )
        torch.Tensor.cumsum = lambda self, *args, **kwargs: ( _cumsum_fix(self, orig_Tensor_cumsum, *args, **kwargs) )
        orig_narrow = torch.narrow
        torch.narrow = lambda *args, **kwargs: ( orig_narrow(*args, **kwargs).clone() )