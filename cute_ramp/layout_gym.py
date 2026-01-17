import cutlass.cute as cute
import torch
import cutlass


@cute.jit
def layout_exp(t: cute.Tensor):
    uo_layout = cute.make_layout((4, 4))
    print(t)
    print(uo_layout)
    print(cute.composition(t, uo_layout))

    smem = cutlass.utils.SmemAllocator()
    sdX = smem.allocate_tensor(cute.BFloat16, cute.make_layout((4, 4)), byte_alignment=16)
    print(sdX)


if __name__ == "__main__":
    a = torch.arange(4 * 4, device="cuda")
    layout_exp(cute.runtime.from_dlpack(a))
