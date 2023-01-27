import sys
import contextlib
import torch
from modules import errors

from modules.accelerators.cuda_accelerator import CudaAccelerator
from modules.accelerators.mps_accelerator import MPSAccelerator
from modules.accelerators.one_api_accelerator import OneApiAccelerator

if sys.platform == "darwin":
    from modules import mac_specific

accelerator = None
# List in priority order; first found will be used
supported_accelerators = [OneApiAccelerator, MPSAccelerator, CudaAccelerator]

for impl in supported_accelerators:
    accelerator = impl.discover()
    if accelerator is not None:
        break

def accelerated():
    return accelerator is not None

def amp():
    return accelerator.amp()

def optimize(model, dtype):
    return accelerator.optimize(model, dtype)

def memory_stats(device=None):
    return accelerator.memory_stats(device)

def memory_summary():
    return accelerator.memory_summary()

def reset_peak_memory_stats():
    return accelerator.reset_peak_memory_stats()

def get_free_memory():
    return accelerator.get_free_memory()

def get_total_memory():
    return accelerator.get_total_memory()

def empty_cache():
    return accelerator.empty_cache()

def manual_seed(seed):
    if accelerated():
        accelerator.manual_seed(seed)
    else:
        torch.manual_seed(seed)

def einsum_op(q, k, v):


    # Smaller slices are faster due to L2/L3/SLC caches.
    # Tested on i7 with 8MB L3 cache.
    return einsum_op_tensor_mem(q, k, v, 32)

def gc():
    accelerator.gc()

def enable_tf32():
    return accelerator.enable_tf32()


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_optimal_device_name():
    accelerator_device = accelerator.get_device()
    if accelerator_device is not None:
        return accelerator_device

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    from modules import shared

    if task in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()

errors.run(accelerator.enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


def randn(seed, shape):
    manual_seed(seed)
    return accelerator.randn(shape)

def randn_without_seed(shape):
    return accelerator.randn(shape)


def autocast(disable=False):
    from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return accelerator.autocast(dtype)


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    from modules import shared

    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."

    message += " Use --disable-nan-check commandline argument to disable this check."

    if device.type == "xpu":
        print(message)
    else:
        raise NansException(message)
