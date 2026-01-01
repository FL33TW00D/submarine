import torch
import torch.nn.functional as F
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)

compiled_fn = torch.compile(
    lambda: F.layer_norm(x, (x.shape[-1],)), mode="max-autotune-no-cudagraphs"
)

for _ in range(3):
    compiled_fn()
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
compiled_fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
