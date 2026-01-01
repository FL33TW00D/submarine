import torch
import torch.nn.functional as F

x = torch.randn((2, 3, 4)) * 10
print(x)

y = F.layer_norm(x, x.shape)
print(y)
