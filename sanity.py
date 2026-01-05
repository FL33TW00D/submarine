import torch
from submarine import our_ln
import triton
import torch.nn.functional as F

DEVICE = triton.runtime.driver.active.get_active_torch_device()


torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = our_ln(x)
y_torch = F.layer_norm(x, (x.shape[-1],))
assert torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-5), (y_triton, y_torch)
