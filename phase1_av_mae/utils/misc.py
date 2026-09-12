import torch
# Snippet 1: Seeding — call this at the top of every script
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Snippet 2: AverageMeter — tracks running mean of any scalar (loss, lr)
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Snippet 3: Load YAML config
from omegaconf import OmegaConf
def load_config(path: str):
    return OmegaConf.load(path)