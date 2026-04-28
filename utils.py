import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# set seed for all random processes
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


# set cuda device when available
def set_device(device_id=None):
    device = None
    if not torch.cuda.is_available():
        print("No GPU available, using CPU.")
        device = "cpu"
    elif device_id is None:
        device = "cuda"
    else:
        device = f"cuda:{device_id}"
    print(f"Using {device} device")
    return device


# set global plot parameters
def configure_plot_parameters(fonts=["Times New Roman", "SimSun"], fontsize=12):
    plt.rcParams["font.family"] = fonts
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.weight"] = "normal"
