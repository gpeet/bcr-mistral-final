"""Seed every random number generator a training script uses."""
import random

import numpy as np


def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU and GPU) so a run can be repeated.

    GPU kernels can still be nondeterministic (e.g. atomic adds on ROCm/CUDA), so
    repeated runs agree to within floating-point noise rather than bit for bit.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
