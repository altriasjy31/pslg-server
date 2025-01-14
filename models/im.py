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

import x_transformers as xf
from x_transformers import Encoder
import flash_pytorch as fh
from flash_pytorch import GAU

class IMEncoder(nn.Module):
  """
  InterPro and Multiple Sequence Alignment Encoder
  """

  # config the parser
  @staticmethod
  def add_model_specific_args(parent_parser):
    """
    Specify the hyperparams of this model
    enc_embed_dims: the embedding dimension of encoder
    enc_depth: the depth of encoder
    enc_num_heads: the number of heads in encoder
    enc_ff_mult: the multiplier of feedforward in encoder
    enc_ff_glu: whether to use GLU in feedforward of encoder
    enc_ff_swish: whether to use swish in feedforward of encoder
    """
    parser = parent_parser.add_argument_group("IMEncoder")
    parser.add_argument("--enc_embed_dims", type = int, default = 128)
    parser.add_argument("--enc_depth", type = int, default = 6)
    parser.add_argument("--enc_num_heads", type = int, default = 8)
    parser.add_argument("--enc_ff_mult", type = int, default = 4)
    parser.add_argument("--enc_ff_glu", action = "store_true")
    parser.add_argument("--enc_ff_swish", action = "store_true")
    return parent_parser


  def __init__(self, embed_dims: int,
               num_classes: int,
               ipr_features: int = 256,
               max_iprs: int = 8,
               msa_features: int = 2048,
               msa_groups: int = 8, # devide msa feature into n grups
               **kwargs) -> None:
    super().__init__()

    self.embed_dims = embed_dims
    self.num_classes = num_classes
    self.ipr_features = ipr_features
    self.msa_features = msa_features

    # lazy linear as projection
    self.msa_groups = msa_groups
    assert msa_features % msa_groups == 0
    self.msa_proj = nn.Linear(msa_features // msa_groups, embed_dims)

    # get the config of the encoder
    depth = kwargs.get('depth', 1)
    num_heads = kwargs.get('num_heads', 8)
    ff_mult = kwargs.get('ff_mult', 4)

    # encoder for ipr and msa
    ff_glu = kwargs.get('ff_glu', False)
    ff_swish = kwargs.get('ff_swish', False)

    pre_norm = False
    self.encoder = Encoder(dim = embed_dims,
                           depth = depth,
                           heads = num_heads,
                           pre_norm = pre_norm,
                           residual_attn = True,    # add residual attention
                           ff_mult = ff_mult,
                           ff_swish = ff_swish,
                           ff_glu = ff_glu,
                           )
    
    self.ln = nn.LayerNorm(embed_dims) if not pre_norm else nn.Identity()
    out_features = (ipr_features*max_iprs) + msa_features
    self.head = nn.Linear(out_features, num_classes)
  
  # override the to, cuda, cpu method
  # save the device
  def to(self, device):
    self.device = device
    return super().to(device)
  
  def cuda(self, device = None):
    if device:
      self.device = device
    else:
      self.device = torch.device('cuda')
    return super().cuda(device)
  
  def cpu(self):
    self.device = torch.device('cpu')
    return super().cpu()
  
  def forward(self, x_ipr: th.Tensor, 
              x_msa: th.Tensor) -> th.Tensor:
    """
    Args:
    x_ipr: (batch, max_iprs, init_ipr_features)
    x_msa: (batch, init_msa_features, h, w)
    """
    size_msa = x_msa.size()

    # (batch_size, init_msa_features)
    x_msa = F.adaptive_avg_pool2d(x_msa, 1).squeeze((2,3))

    # group
    x_msa = x_msa.view(size_msa[0], self.msa_groups, -1) # batrch_size, msa_groups, ...
   
    # project
    # x_ipr = self.ipr_proj(x_ipr)
    x_msa = self.msa_proj(x_msa)

    # concat
    x = th.cat([x_ipr, x_msa], dim = 1)

    # encoder
    x = self.ln(self.encoder(x))

    assert isinstance(x, th.Tensor)

    return self.head(x.flatten(1))

class Residual(nn.Module):
  """
  Residual connection
  """
  def __init__(self, module: nn.Module) -> None:
    super().__init__()
    self.module = module
  
  def forward(self, x: th.Tensor) -> th.Tensor:
    return x + self.module(x)

# based on GAU
class IMGAU(nn.Module):
  """
  InterPro and Multiple Sequence Alignment Encoder (Gated Attention Unit)
  """
  def __init__(self, embed_dims: int,
               num_classes: int, **kwargs) -> None:
    super().__init__()

    self.embed_dims = embed_dims
    self.num_classes = num_classes

    # lazy linear as projection
    self.ipr_proj = nn.LazyLinear(embed_dims)
    self.msa_proj = nn.LazyLinear(embed_dims)

    # get config of the encoder (GAU)
    query_key_dim = kwargs.get('query_key_dim', 128)
    expansion_factor = kwargs.get('expansion_factor', 2) # multiplies the embed_dims
    laplace_attn_fn = kwargs.get('laplace_attn_fn', True)
    depth = kwargs.get('depth', 6)

    # build the encoder
    # n_depth of LayerNorm(GAU(x)), GAU contains residual
    self.encoder = \
      nn.ModuleList([nn.Sequential(
        GAU(dim = embed_dims,
                      query_key_dim = query_key_dim,
                      expansion_factor = expansion_factor,
                      laplace_attn_fn = laplace_attn_fn),
        nn.LayerNorm(embed_dims))
        for _ in range(depth)])
    
    self.head = nn.Linear(embed_dims, num_classes)
  
  # override the to, cuda, cpu method to save the device
  def to(self, device):
    self.device = device
    return super().to(device)
  
  def cuda(self, device = None):
    if device:
      self.device = device
    else:
      self.device = torch.device('cuda')
    return super().cuda(device)
  
  def cpu(self):
    self.device = torch.device('cpu')
    return super().cpu()
  
  def forward(self, x_ipr: th.Tensor,
              x_msa: th.Tensor):
    """
    Args:
    x_ipr: (batch, max_iprs, init_ipr_features)
    x_msa: (batch, max_msa, init_msa_features)
           or (batch, init_msa_features, h, w)
    """
    size_msa = x_msa.size()
    if len(size_msa) == 4:
      x_msa = x_msa.flatten(2).transpose(1, 2)

    # project
    x_ipr = self.ipr_proj(x_ipr)
    x_msa = self.msa_proj(x_msa)

    # concat
    x = th.cat([x_ipr, x_msa], dim = 1)

    # encoder
    for layer in self.encoder:
      x = layer(x)
    
    return self.head(x.mean(1))