import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cute_viz import render_tv_layout_svg


@cute.jit
def visualize_tv_layout(
    mA: cute.Tensor,
):
    tile_mn = (8, 8)
    tv_layout = cute.make_layout(shape=((2, 2, 2), (2, 2, 2)), stride=((1, 16, 4), (8, 2, 32)))

    render_tv_layout_svg(tv_layout, tile_mn, "./tv_layout.svg")


if __name__ == "__main__":
    M, N = 24, 1
    a = torch.arange(M * N, device="cuda", dtype=torch.int32).reshape(M, N)
    a_ = from_dlpack(a, assumed_align=16)

    visualize_tv_layout(a_)
