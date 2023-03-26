import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

def override_func(inst, func, func_name):
    bound_method = func.__get__(inst, inst.__class__)
    setattr(inst, func_name, bound_method)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def gen_folder_name(args):
    def get_attr(inst, arg):
        value = getattr(inst, arg)
        if isinstance(value, float):
            return f"{value:.4f}"
        else:
            return value
    folder_name = ''
    for arg in vars(args):
        folder_name += f'{arg}-{get_attr(args, arg)}~'
    return folder_name[:-1]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count