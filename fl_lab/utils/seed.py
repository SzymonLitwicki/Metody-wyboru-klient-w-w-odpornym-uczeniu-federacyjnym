import os
import random
import zlib

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def derive_seed(master: int, subsystem: str) -> int:
    key = f"{master}-{subsystem}".encode()
    return zlib.crc32(key) & 0xFFFFFFFF
