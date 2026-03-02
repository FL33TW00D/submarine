import triton
from triton.runtime import driver

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEVICE = triton.runtime.driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)

NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

# Get CUDA / compiler info
import torch

cuda_version = torch.version.cuda or "N/A"
cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
device_name = torch.cuda.get_device_name(DEVICE.index)
compute_cap = torch.cuda.get_device_capability(DEVICE.index)
total_mem = torch.cuda.get_device_properties(DEVICE.index).total_memory / (1024**3)

BAR = f"{DIM}{'─' * 52}{RESET}"

print()
print(f"{BOLD}{MAGENTA}  ╔══════════════════════════════════════════════════╗{RESET}")
print(f"{BOLD}{MAGENTA}  ║            ⚡  GPU & CUDA INFO  ⚡               ║{RESET}")
print(f"{BOLD}{MAGENTA}  ╚══════════════════════════════════════════════════╝{RESET}")
print()

print(f"  {BOLD}{CYAN}DEVICE{RESET}")
print(f"  {BAR}")
print(f"  {GREEN}{'Name':<28}{RESET}{device_name}")
print(f"  {GREEN}{'Compute Capability':<28}{RESET}{compute_cap[0]}.{compute_cap[1]}")
print(f"  {GREEN}{'Total Memory':<28}{RESET}{total_mem:.1f} GB")
print()

print(f"  {BOLD}{CYAN}SOFTWARE{RESET}")
print(f"  {BAR}")
print(f"  {GREEN}{'CUDA Version':<28}{RESET}{cuda_version}")
print(f"  {GREEN}{'cuDNN Version':<28}{RESET}{cudnn_version}")
print(f"  {GREEN}{'Triton Version':<28}{RESET}{triton.__version__}")
print(f"  {GREEN}{'PyTorch Version':<28}{RESET}{torch.__version__}")
print()

print(f"  {BOLD}{CYAN}HARDWARE PROPERTIES{RESET}")
print(f"  {BAR}")
print(f"  {YELLOW}{'Streaming Multiprocessors':<28}{RESET}{NUM_SM}")
print(f"  {YELLOW}{'Max Registers per Block':<28}{RESET}{NUM_REGS:,}")
print(f"  {YELLOW}{'Shared Memory per SM':<28}{RESET}{SIZE_SMEM:,} bytes ({SIZE_SMEM / 1024:.0f} KB)")
print(f"  {YELLOW}{'Warp Size':<28}{RESET}{WARP_SIZE}")
print()
