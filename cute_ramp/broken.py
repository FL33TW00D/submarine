import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.jit
def print_crd_for_idx(shape, stride):
    layout = cute.make_layout(shape, stride=stride)
    cute.printf(layout)
    for idx in range(cute.size(shape)):
        crd = layout.get_hier_coord(idx)
        cute.printf("{} -> {}", idx, crd)


print_crd_for_idx((4, 4), (4, 1))
