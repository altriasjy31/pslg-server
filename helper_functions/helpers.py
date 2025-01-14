from copy import deepcopy

import typing as T
from typing import List, OrderedDict
import numpy as np
import numpy.typing as nT
import torch
import torch.nn as nn
import torch.nn.parallel as dp
import yaml
from functools import partial, reduce

import argparse as P
from helper_functions.metrics import AuPRC_score, fmax_score

Tensor = torch.Tensor
NDarray = nT.NDArray
ndarray = np.ndarray

def add_config(opt: P.Namespace):
  assert opt is not None
  with open(opt.config, "r") as h:
    config_dict: T.Dict = yaml.safe_load(h)
  assert "base" in config_dict, \
      "we firstly get the base configuration filepath from the value of 'base' key, " +\
          "then, read the common option from the base configuration file"
  with open(config_dict["base"], "r") as h:
    base_dict: T.Dict = yaml.safe_load(h)
  option_dict = opt.__dict__
  option_dict.update(base_dict)
  option_dict.update(config_dict) # set the num classes of config_dict and update unique options
  
  return option_dict


def load_by_module_name(cpu_model: nn.Module,
                        gpu_model: nn.Module,
                        r: None, 
                        module_name: str):
    """
    Using:
    partial_loading = partial(load_by_module_name, cpu_model, gpu_model)
    reduce(partial_loading, module_names, None)
    """
    gpu_module = gpu_model.get_submodule(module_name)
    if isinstance(gpu_module, dp.DataParallel):
        gpu_module = gpu_module.module
    else:
        pass
    
    cpu_model.get_submodule(module_name).load_state_dict(gpu_module.state_dict())
    
    return r
    
def loading_with_cpu(cpu_model: nn.Module, gpu_model: nn.Module, weights: OrderedDict,
                     module_names: List[str]):
    """
    loading weights that trained on gpu into cpu
    """
    gpu_model.load_state_dict(weights)
    partial_loading = partial(load_by_module_name, cpu_model, gpu_model)
    reduce(partial_loading, module_names, None)

def check_model_weights(model0: nn.Module, model1: nn.Module):
    return all([torch.equal(w0,w1) 
                for w0, w1 in zip(model0.state_dict().values(),
                                  model1.state_dict().values())])

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = (self.ema if self.ema is not None 
                             else self.val) * 0.99 + self.val * 0.01


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


if __name__ == "__main__":
    """
    """
    targs = np.array([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = np.array([[0.2, 0.3, 0.3, 0.5], [0.2, 0.3, 0.5, 0.3]])
    f0 = fmax_score(targs, preds)
    a0 = AuPRC_score(targs, preds) 
    print("fmax {:.4f}, AuPRC {:.4f}".format(f0,a0))
