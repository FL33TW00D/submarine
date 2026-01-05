import torch

eps = 1e-5


class LayerNorm:
    @staticmethod
    def forward(x, w, b):
        # x is the input activations, of shape B,T,C
        # w are the weights, of shape C
        # b are the biases, of shape C
        B, T, C = x.size()
        # calculate the mean
        mean = x.sum(-1, keepdim=True) / C  # B,T,1
        # calculate the variance
        xshift = x - mean  # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C  # B,T,1
        # calculate the inverse standard deviation: **0.5 is sqrt, **-0.5 is 1/sqrt
        rstd = (var + eps) ** -0.5  # B,T,1
        # normalize the input activations
        norm = xshift * rstd  # B,T,C
        # scale and shift the normalized activations at the end
        out = norm * w + b  # B,T,C

        # return the output and the cache, of variables needed later during the backward pass
        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = (
            dnorm
            - dnorm.mean(-1, keepdim=True)
            - norm * (dnorm * norm).mean(-1, keepdim=True)
        )
        dx *= rstd
        return dx, dw, db


B = 2  # some toy numbers here
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)

dout = torch.randn(B, T, C)
fakeloss = (out * dout).sum()
fakeloss.backward()

dx, dw, db = LayerNorm.backward(dout, cache)
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())
