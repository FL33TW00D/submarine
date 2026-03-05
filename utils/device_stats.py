import triton
from triton.runtime import driver
import torch

# ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEVICE = triton.runtime.driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

# torch device properties (richer info)
tp = torch.cuda.get_device_properties(DEVICE.index)
cuda_version = torch.version.cuda or "N/A"
cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
device_name = torch.cuda.get_device_name(DEVICE.index)
compute_cap = torch.cuda.get_device_capability(DEVICE.index)
total_mem = tp.total_memory / (1024**3)

# Memory stats
mem_allocated = torch.cuda.memory_allocated(DEVICE.index) / (1024**2)
mem_reserved = torch.cuda.memory_reserved(DEVICE.index) / (1024**2)
mem_free = (tp.total_memory - torch.cuda.memory_allocated(DEVICE.index)) / (1024**3)

# Additional properties from torch
l2_cache_size = tp.L2_cache_size  # bytes
total_mem_bw = tp.total_memory  # we'll compute bandwidth below
clock_rate_mhz = tp.clock_rate / 1000 if hasattr(tp, "clock_rate") else None
mem_clock_mhz = tp.memory_clock_rate / 1000 if hasattr(tp, "memory_clock_rate") else None
mem_bus_width = tp.memory_bus_width if hasattr(tp, "memory_bus_width") else None

# Compute theoretical memory bandwidth (GB/s): 2 * mem_clock (Hz) * bus_width (bits) / 8 / 1e9
if mem_clock_mhz and mem_bus_width:
    mem_bandwidth_gbs = 2 * (mem_clock_mhz * 1e6) * mem_bus_width / 8 / 1e9
else:
    mem_bandwidth_gbs = None

# Cores per SM heuristic based on compute capability
cores_per_sm_map = {
    (7, 0): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,
    (9, 0): 128,
}
cores_per_sm = cores_per_sm_map.get(compute_cap, None)
total_cuda_cores = NUM_SM * cores_per_sm if cores_per_sm else None

# Theoretical FLOPS (FP32): cores * 2 (FMA) * clock (Hz)
if total_cuda_cores and clock_rate_mhz:
    tflops_fp32 = total_cuda_cores * 2 * (clock_rate_mhz * 1e6) / 1e12
else:
    tflops_fp32 = None

BAR = f"{DIM}{'─' * 52}{RESET}"


def row(label, value, color=GREEN):
    print(f"  {color}{label:<32}{RESET}{value}")


print()
print(f"{BOLD}{MAGENTA}  ╔══════════════════════════════════════════════════╗{RESET}")
print(f"{BOLD}{MAGENTA}  ║            ⚡  GPU & CUDA INFO  ⚡               ║{RESET}")
print(f"{BOLD}{MAGENTA}  ╚══════════════════════════════════════════════════╝{RESET}")

print(f"\n  {BOLD}{CYAN}DEVICE{RESET}")
print(f"  {BAR}")
row("Name", device_name)
row("Compute Capability", f"{compute_cap[0]}.{compute_cap[1]}")
row("Total Memory", f"{total_mem:.1f} GB")
row("Multi-GPU (is_multi_gpu)", tp.is_multi_gpu_board if hasattr(tp, "is_multi_gpu_board") else "N/A")

print(f"\n  {BOLD}{CYAN}SOFTWARE{RESET}")
print(f"  {BAR}")
row("CUDA Version", cuda_version)
row("cuDNN Version", cudnn_version)
row("Triton Version", triton.__version__)
row("PyTorch Version", torch.__version__)
row("CUDA Arch List", torch.cuda.get_arch_list()[-3:] if torch.cuda.get_arch_list() else "N/A")

print(f"\n  {BOLD}{CYAN}COMPUTE{RESET}")
print(f"  {BAR}")
row("Streaming Multiprocessors", NUM_SM, YELLOW)
if total_cuda_cores:
    row("CUDA Cores (estimated)", f"{total_cuda_cores:,} ({cores_per_sm}/SM)", YELLOW)
if clock_rate_mhz:
    row("GPU Clock Rate", f"{clock_rate_mhz:,.0f} MHz", YELLOW)
row("Max Registers per Block", f"{NUM_REGS:,}", YELLOW)
row("Max Threads per SM", tp.max_threads_per_multi_processor, YELLOW)
row("Max Threads per Block", tp.max_threads_per_block if hasattr(tp, "max_threads_per_block") else "N/A", YELLOW)
row("Warp Size", WARP_SIZE, YELLOW)
if tflops_fp32:
    row("Peak FP32 (theoretical)", f"{tflops_fp32:.1f} TFLOPS", YELLOW)

print(f"\n  {BOLD}{CYAN}MEMORY HIERARCHY{RESET}")
print(f"  {BAR}")
row("L2 Cache Size", f"{l2_cache_size:,} bytes ({l2_cache_size / (1024**2):.1f} MB)", RED)
row("Shared Memory per SM", f"{SIZE_SMEM:,} bytes ({SIZE_SMEM / 1024:.0f} KB)", RED)
if mem_clock_mhz:
    row("Memory Clock Rate", f"{mem_clock_mhz:,.0f} MHz", RED)
if mem_bus_width:
    row("Memory Bus Width", f"{mem_bus_width} bits", RED)
if mem_bandwidth_gbs:
    row("Memory Bandwidth (theor.)", f"{mem_bandwidth_gbs:,.0f} GB/s", RED)

print(f"\n  {BOLD}{CYAN}MEMORY USAGE{RESET}")
print(f"  {BAR}")
row("Allocated", f"{mem_allocated:.1f} MB")
row("Reserved (cached)", f"{mem_reserved:.1f} MB")
row("Free (approx)", f"{mem_free:.2f} GB")

print()
