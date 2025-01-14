import typing as T
import os
import pathlib
import sys


prj_dir = str(pathlib.Path(__file__).parent.parent)
if prj_dir not in sys.path:
  sys.path.append(prj_dir)

# from .heads import GroupLinear
# from .heads import torch_interpro_mlp

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy.random as nrand
import pandas as pd
import sklearn.preprocessing as prep
# import xformers.factory.model_factory as xf
# import xformers.components.activations as xa
# import xformers.components.feedforward as ff
# import xformers.triton as xt
# import xformers.ops.swiglu_op as xsw
import functools as ft
import dataclasses as ds

class IPRFormer(nn.Module):
  """
  Args:
    interpros: a dataframe with columns: protein, interpros
    embed_dims: the embedding dimension
    num_classes: the number of classes
    ipr_groups: the number of groups for the interpros in projection
    **kwargs: other arguments, e.g., max_interpros: the max number of interpros per protein
  """
  def __init__(self, embed_dims: int,
               num_interpros: int,
               num_classes: int,
               ipr_groups: int | None = 8, 
               **kwargs) -> None:
    super().__init__()

    self.num_interpros = num_interpros

    self.embed_dims = embed_dims
    self.num_classes = num_classes
    self.ipr_groups = ipr_groups

    if ipr_groups is None: ipr_groups = 1

    assert embed_dims % ipr_groups == 0

    max_iprs = kwargs.get("max_interpros", 100) # max number of interpros per protein, ipr_groups + max_tokens
    # get device
    # kwargs has gpu_ids, which is a list of int
    # if gpu_ids is not None, then use gpu_ids[0]
    # else use cpu
    gpu_ids = kwargs.get("gpu_ids", None)
    self.device = th.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")

    self.max_iprs = max_iprs
    dim_model = embed_dims // ipr_groups
    self.dim_model = dim_model

    # ipr_embed flag
    use_ipr_embed = kwargs.get("use_ipr_embed", False)

    if use_ipr_embed:
      self.ipr_embed = nn.Embedding(num_interpros + 1, dim_model)
    else:
      self.ipr_embed = None

    if ipr_groups > 1:
      # stacked = True
      # input_dims = num_interpros
      # self.init_ipr_proj = GroupLinear(input_dims, embed_dims,
      #                                   ipr_groups, torch_interpro_mlp,
      #                                   stacked=stacked)
      pass
    else:
     self.init_ipr_proj = nn.Linear(num_interpros, embed_dims)

    self.head = nn.Linear(embed_dims, num_classes)
  
  def to(self, device):
    self.device = device
    return super().to(device)
  
  def cuda(self):
    self.device = th.device("cuda:0")
    return super().cuda()
  
  def cpu(self):
    self.device = th.device("cpu")
    return super().cpu()
  
  def _preprose_interpros(self, x: th.Tensor) -> th.Tensor:
    # x is a list of interpros (with padding)
    # binarize
    # embed
    # using self.init_ipr_proj to project the binarized interpros
    # concat the projections and embeddings
    # return the concatenated tensor

    # embed
    # shape: (batch_size, max_tokens, dim_model)
    if self.ipr_embed:
      x_embed = self.ipr_embed(x)
    else:
      x_embed = None
    # binarize -> (batch_size, num_interpros+1)
    batch_size = x.size(0)
    # note: need include padding index
    bipr = th.zeros(batch_size, self.num_interpros + 1,
                    dtype=x.dtype,
                    device=x.device,)
    bipr[th.arange(batch_size).unsqueeze(1), x] = 1
    # remove padding index
    bipr = bipr[:, :-1]
    bipr = bipr.float()

    # proj
    # shape: (batch_size, ipr_groups, dim_model)
    x_proj = self.init_ipr_proj(bipr)
    # return the concatenated tensor
    if x_embed:
      return th.concat([x_embed, x_proj], dim=1) # shape: (batch_size, max_iprs, dim_model)
    else:
      return x_proj
  
  def forward(self, x) -> th.Tensor:
    # x is a list of protein names
    # preprocess the interpros
    # feed into the transformer
    # feed into the head
    # return the logits
    assert hasattr(self, "device"), "Please set the device first"
    ipr_embed = self._preprose_interpros(x)
    return self.head(ipr_embed.flatten(1))
  
  def forward_features(self, x) -> th.Tensor:
    # x is a list of protein names
    # preprocess the interpros
    # feed into the transformer
    # return the transformer output
    ipr_embed = self._preprose_interpros(x)
    return ipr_embed
