import torch
import torch.nn.functional as F
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_torch = F.layer_norm(x, (x.shape[-1],))
