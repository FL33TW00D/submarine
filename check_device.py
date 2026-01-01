import triton
import torch
import torch.nn.functional as F
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
print("DEVICE PROPERTIES")
print("NUM SM: ", NUM_SM)
print("NUM REG: ", NUM_REGS)
print("SIZE SMEM PER SM: ", SIZE_SMEM)
print("WARP SIZE: ", WARP_SIZE)
