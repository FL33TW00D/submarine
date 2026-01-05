import torch


def measure_bandwidth():
    x = torch.randn(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    y.copy_(x)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    gb = 2 * x.numel() * 4 / 1e9
    print(f"Measured bandwidth: {gb / (ms / 1000):.1f} GB/s")


measure_bandwidth()
