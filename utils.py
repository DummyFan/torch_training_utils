import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def set_device():
    # GPU可用时优先调用GPU加速训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')
    return device
