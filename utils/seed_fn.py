import torch
import numpy as np
import random
import os
def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(num)
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)