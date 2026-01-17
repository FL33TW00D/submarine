import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

"""
Proper 2D TV layout example.

24-element array, 4 threads, each thread owns 6 values.

Visualize as 6x4 matrix:
     col0 col1 col2 col3
row0:  0    1    2    3
row1:  4    5    6    7
row2:  8    9   10   11
row3: 12   13   14   15
row4: 16   17   18   19
row5: 20   21   22   23

Thread assignment (2x2 grid of threads):
     col0 col1 col2 col3
row0: T0   T0   T1   T1
row1: T0   T0   T1   T1
row2: T0   T0   T1   T1
row3: T2   T2   T3   T3
row4: T2   T2   T3   T3
row5: T2   T2   T3   T3

Each thread owns a 3x2 tile of values:
  T0: [0,1], [4,5], [8,9]
  T1: [2,3], [6,7], [10,11]
  T2: [12,13], [16,17], [20,21]
  T3: [14,15], [18,19], [22,23]
"""


@cute.kernel
def demo_kernel(data: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()

    # Compose data with TV layout
    partitioned = cute.composition(data, tv_layout)

    # Each thread asks for its values
    my_values = partitioned[(tidx, None)]

    cute.printf("Thread {} owns: {}", tidx, my_values)


@cute.jit
def run_demo(data: cute.Tensor):
    # Thread layout: (2, 2) : (2, 12)
    #   - 2x2 grid of threads
    #   - Stride 2: T0->T1 (down in thread grid) = +2 in memory
    #   - Stride 12: T0->T2 (right in thread grid) = +12 in memory
    #
    # Value layout: (2, 3) : (1, 4)
    #   - Each thread owns 2x3 values
    #   - Stride 1: consecutive in first mode
    #   - Stride 4: jump by 4 in second mode (next row of the 6x4 matrix)

    tv_layout = cute.make_layout(
        shape=((2, 2), (2, 3)),  # (thread_shape, value_shape)
        stride=((2, 12), (1, 4)),  # (thread_stride, value_stride)
    )

    print(f"TV Layout: {tv_layout}")
    print(f"Thread layout: (2,2):(2,12) -> 4 threads")
    print(f"Value layout:  (2,3):(1,4)  -> 6 values per thread")
    print(f"Total elements: {cute.size(tv_layout)}")
    print()

    num_threads = cute.size(tv_layout, mode=[0])  # Size of thread mode
    print(f"Launching {num_threads} threads")

    demo_kernel(data, tv_layout).launch(
        grid=(1, 1, 1),
        block=(num_threads, 1, 1),
    )


if __name__ == "__main__":
    # 24 elements: [0, 1, 2, ..., 23]
    data = torch.arange(24, device="cuda", dtype=torch.int32)
    data_ = from_dlpack(data, assumed_align=16)

    print("Linear array: [0, 1, 2, ..., 23]")
    print()
    print("Conceptually a 6x4 matrix:")
    print(data.reshape(6, 4))
    print()

    run_demo(data_)
    torch.cuda.synchronize()
