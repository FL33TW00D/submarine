import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
import cutlass


# Inner partition pure
# @cute.jit
# def launch_copy(
#    tensor_S: cute.Tensor,
#    tensor_D: cute.Tensor,
#    block_shape: cute.Shape,
#    num_threads: cutlass.Constexpr[cutlass.Int32],
# ):
#    print("Tensors:")
#    print(f"tensor_S = {tensor_S}")
#    print(f"tensor_D = {tensor_D}")
#
#    # Tile (m, n) by (M, N) to obtain ((M, N), m', n')
#    # , where M' and N' are the number of block tiles
#    tiled_tensor_S = cute.tiled_divide(tensor_S, block_shape)  # (M, N), m', n')
#    tiled_tensor_D = cute.tiled_divide(tensor_D, block_shape)  # (M, N), m', n')
#
#    print("Block Tile Tensor:")
#    print(f"tiled_tensor_S = {tiled_tensor_S}")
#    print(f"tiled_tensor_D = {tiled_tensor_D}")


# Inner and outer partition
@cute.jit
def launch_copy(
    tensor_S: cute.Tensor,  # Pointer to Source
    tensor_D: cute.Tensor,  # Pointer to Destination
    block_shape: cute.Shape,  # (M, N)
    thread_shape: cutlass.Shape,
):
    print("Tensors:")
    print(f"tensor_S = {tensor_S}")
    print(f"tensor_D = {tensor_D}")

    # Tile (m, n) by (M, N) to obtain ((M, N), (m', n'))
    # , where M' and N' are the number of block tiles
    tiled_tensor_S = cute.zipped_divide(tensor_S, block_shape)  # (M, N), (m', n'))
    tiled_tensor_D = cute.zipped_divide(tensor_D, block_shape)  # (M, N), (m', n'))

    print("Block Tile Tensor:")
    print(f"tiled_tensor_S = {tiled_tensor_S}")
    print(f"tiled_tensor_D = {tiled_tensor_D}")

    thr_layout = cute.make_layout(thread_shape, stride=(thread_shape[1], 1))

    print("Thread Layout:")
    print(f"thr_layout = {thr_layout}")

    grid_dim = (
        cute.size(tiled_tensor_D, mode=[1]),
        1,
        1,
    )
    block_dim = (cute.size(thr_layout), 1, 1)

    print("Grid and Block Configuration:")
    print(f"grid_dim = {grid_dim}")
    print(f"block_dim = {block_dim}")


if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    tensor_shape = (8192, 8192)
    block_shape = (32, 256)
    thread_shape = (8, 32)

    S = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    D = torch.zeros(8192, 8192, device="cuda", dtype=torch.bfloat16)

    tensor_S = from_dlpack(S, assumed_align=16)
    tensor_D = from_dlpack(D, assumed_align=16)

    launch_copy(tensor_S, tensor_D, block_shape, thread_shape)

    print(S)
    print(D)
