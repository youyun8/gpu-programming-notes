# Simple Inference (LeetGPU)
# https://leetgpu.com/challenges/simple-inference
import torch
import torch.nn as nn


# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    # A Linear layer is one GEMM with a fused bias: output = bias + input @ weight^T.
    # Writing with out= avoids a temporary; inference_mode skips autograd bookkeeping.
    with torch.inference_mode():
        if model.bias is None:  # nn.Linear(..., bias=False)
            torch.matmul(input, model.weight.t(), out=output)
        else:
            torch.addmm(model.bias, input, model.weight.t(), out=output)
