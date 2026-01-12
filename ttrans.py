import torch
import triton
import triton.language as tl


@triton.jit
def trans_kernel(ptr):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)

    block_ptr = ptr + (offsets_m[:, None] * 16 + offsets_n[None, :])
    data = tl.load(block_ptr)

    transposed_data = tl.trans(data)

    store_ptr = (ptr + 256) + (offsets_m[:, None] * 16 + offsets_n[None, :])
    tl.store(store_ptr, transposed_data)


x = torch.zeros(512, device="cuda", dtype=torch.float32)
grid = (1,)
kern = trans_kernel[grid](x)

device = torch.cuda.current_device()
cache = trans_kernel.device_caches[device]
compiled_kernel = list(cache[0].values())[0]
print(compiled_kernel.asm["ptx"])
