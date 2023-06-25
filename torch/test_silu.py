import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

bs = 1
seq = 512
hs = 1024
# M = bs * seq
# K = hs
# N = hs * 3
dtype = torch.float16
hidden_states = torch.rand((bs, seq, hs), dtype=dtype, device="cuda:0")


class SiluAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x)


ntest = 100
mod = SiluAct().cuda().to(dtype)


def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time - start_time) * 1e6)
    return times, res


def run_cuda():
    return torch.ops.ft.silu(hidden_states)


def run_torch():
    with torch.no_grad():
        return mod(hidden_states)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.ops.load_library("build/libft.so")
    torch_res = run_torch()
    # it will change the state, be careful!
    cuda_res = run_cuda()
    print(cuda_res)
    print(torch_res)
    torch.testing.assert_close(cuda_res, torch_res)
    print("Kernel test passed.")

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
