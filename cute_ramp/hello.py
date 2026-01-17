import cutlass
import cutlass.cute as cute


@cute.kernel
def hello_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < 10:
        cute.printf("Hello from GPU")


@cute.jit
def hello_world():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


compiled = cute.compile(hello_world)
compiled()
