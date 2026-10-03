import os

import torch

# Without a GPU, run the Triton kernels on CPU through Triton's interpreter.
# This has to happen before torchrolling (and so triton.jit) is imported.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
