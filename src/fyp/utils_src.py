import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)                 # Python's RNG
    np.random.seed(seed)              # NumPy's RNG
    torch.manual_seed(seed)           # CPU + all CUDA devices
    torch.cuda.manual_seed_all(seed)  # explicit CUDA seeding, redundant

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)