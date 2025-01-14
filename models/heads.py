import typing as T
import os
import pathlib
import sys
from typing import Optional

# from xformers.components import Activation

prj_dir = str(pathlib.Path(__file__).parent.parent)
if prj_dir not in sys.path:
  sys.path.append(prj_dir)

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
# import xformers.ops.swiglu_op as xsw

import x_transformers.x_transformers as xtt
from x_transformers import Encoder

from flash_pytorch import GAU

import functools as ft
import dataclasses as ds

# @ds.dataclass
# class SwiGLUConfig(ff.FeedforwardConfig):
#   hidden_layer_multiplier: int
#   bias: bool

# @ff.register_feedforward("SwiGLU", SwiGLUConfig)
# class SwiGLU(ff.Feedforward):
#   def __init__(self, dim_model: int, 
#                hidden_layer_multiplier: int,
#                bias: bool = True,
#                *args, **kwargs):
#     super().__init__()
#     dim_mlp = hidden_layer_multiplier * dim_model
#     self.mlp = xsw.SwiGLU(dim_model, dim_mlp,
#                           bias=bias)
#   def forward(self, inputs: th.Tensor):
#     return self.mlp(inputs)

class IPR(nn.Module):
  proteins: T.List[str] | None = None
  # call this method, before forward, if need interprot intput
  def set_proteins(self, proteins: T.Sequence[str] | None):
    if proteins is None:
      self.proteins = None
    else:
      self.proteins = proteins if isinstance(proteins, T.List) else list(proteins)
  
  def get_proteins(self):
    return self.proteins

def interpro_encoding(ipr,
                      interpro_index: T.Dict[str, int],
                      max_interpro_size: int = 100,
                      device: th.device = th.device("cuda:0")):
  # (batch_size, num_interpros)
  ipr = [[interpro_index[i] for i in xs] for xs in ipr] # to index
  # (batch_size, ipr_size)
  # not using shuffle
  return th.concat([F.pad(th.tensor(xs), (0, max_interpro_size - len(xs)))[None, :] 
                  if len(xs) < max_interpro_size
                  else th.tensor(xs)[None, :max_interpro_size]
                  for xs in ipr], dim=0).to(device)

# class InterProHead(IPR):
#   def __init__(self, interpros: pd.DataFrame, # size: (num_proteins, num_interpros)
#                embed_dims: int,
#                num_classes: int,
#                ipr_groups: T.Union[int, None] = 16,
#                msa_groups: T.Union[int, None] = None,
#                **kwargs,
#                ) -> None:
#     super().__init__()

#     assert "interpros" in interpros.columns
#     self.interpros = interpros
#     self.mlb = prep.MultiLabelBinarizer().fit(self.interpros["interpros"])
#     num_interpros = self.mlb.classes_.shape[0]
#     self.num_interpros = num_interpros
#     self.embed_dims = embed_dims
#     self.num_classes = num_classes
#     msa_embed_dims = kwargs.get("msa_embed_dims", 2048)

#     if ipr_groups is None: ipr_groups = 1
#     if msa_groups is None: msa_groups = 1

#     assert embed_dims % ipr_groups == 0
#     assert msa_embed_dims % msa_groups == 0

#     if ipr_groups > 1:
#       stacked = True
#       num_layers = 6
#       num_heads = 8
#       dropout = 0.1
#       activation = "gelu"
#       self.init_ipr_proj = GroupLinear(num_interpros, embed_dims,
#                                         ipr_groups, torch_interpro_mlp,
#                                         stacked=stacked)
#       xformers_config = xf.xFormerConfig(
#         [
#           {
#             "block_type": "encoder",
#             "dim_model": embed_dims // ipr_groups,
#             "num_layers": num_layers,
#             "residual_norm_style": "deepnorm",
#             "multi_head_config": {
#               "num_heads": num_heads,
#               "residual_dropout": dropout,
#               "use_separate_proj_weight": True,
#               "bias": True,
#               "attention": {
#                   "name": "scaled_dot_product",
#                   "dropout": dropout,
#                   "causal": False,
#                   "seq_len": ipr_groups,
#               },
#               "dim_model": embed_dims // ipr_groups,
#             },
#             "feedforward_config": {
#               "name": "FusedMLP",
#               "dropout": dropout,
#               "activation": activation,
#               "hidden_layer_multiplier": 4,
#               "dim_model": embed_dims // ipr_groups,
#             },
#           },
#         ]
#       )
#       self.transformer = xf.xFormer.from_config(xformers_config)
#       self.ln = nn.LayerNorm(embed_dims // ipr_groups)
#       self.ipr_proj = None
#     else:
#       self.init_ipr_proj = torch_interpro_mlp(num_interpros, embed_dims)
#       self.transformer = nn.Identity()
#       self.ln = nn.Identity() # type: ignore
#       self.ipr_proj = torch_interpro_mlp(embed_dims, embed_dims,
#                                         residual=True)

#     self.init_msa_proj: T.Union[GroupLinear, None]
#     if msa_groups > 1:
#       self.init_msa_proj = GroupLinear(msa_embed_dims, embed_dims,
#                                        msa_groups,
#                                        torch_interpro_mlp)
#     else:
#       self.init_msa_proj = None

#     self.linear = nn.Linear(msa_embed_dims + embed_dims, num_classes)
  
#   def forward(self, x: th.Tensor):
#     # (batch_size, embed_dims, feature_h, feature_w)
#     x = F.adaptive_avg_pool2d(x, 1).squeeze((2,3)) # (batch_size, embed_dims)
#     batch_size = x.size(0)
#     if self.proteins is not None:
#       assert self.proteins is not None
#       ipr = th.tensor(self.mlb.transform(self.interpros.loc[self.proteins, "interpros"])).to(x.device) # type: ignore
#       ipr = ipr.float() # must be float, not be int64
      
#       ipr = self.init_ipr_proj(ipr)
#       ipr = self.ln(self.transformer(ipr))
#       ipr = ipr.view(-1, self.embed_dims)
#       if self.ipr_proj is not None:
#         ipr = self.ipr_proj(ipr) # contains residual structure
#     else:
#       # (batch_size, embed_dims)
#       ipr = th.zeros((batch_size, self.embed_dims), device=x.device).float()
    
#     if self.init_msa_proj is not None:
#       x = self.init_msa_proj(x)
#     return self.linear(th.concat([x, ipr],dim=1)) # embed_dims * 2
  
#   @classmethod
#   def from_file(cls, prot_interpro_file: str,
#                 embed_dims: int,
#                 num_classes: int,
#                 **kwargs,):
#     interpros = pd.read_pickle(prot_interpro_file)
#     return cls(interpros, embed_dims, num_classes, **kwargs)

# # refer to DeepGOZero
# class InterProMLP(nn.Module):
#   def __init__(self, in_features: int, 
#                out_features:int,
#                linear = nn.Linear, 
#                dropout= None, 
#                activation = nn.ReLU(),
#                bias = True,
#                norm = None,
#                residual = False) -> None:
#     super().__init__()

#     # self.linear = nn.Linear(in_features, out_features, bias=bias)
#     self.linear = linear(in_features, out_features, bias=bias)
#     self.activation = activation
#     self.norm = norm
#     self.dropout = dropout
#     self.residual = residual
#   def forward(self, x: th.Tensor):
#     if self.residual:
#       res = x
#     else:
#       res = th.tensor(0, requires_grad=False)
#     x = self.activation(self.linear(x))
#     if self.norm is not None:
#       x = self.norm(x)
    
#     if self.dropout is not None:
#       x = self.dropout(x)
#     return x + res

#   def get_weight(self):
#     return self.linear.weight

# def torch_interpro_mlp(in_features, out_features,
#                        activation=nn.ReLU(),
#                        residual=False):
#   return InterProMLP(in_features, out_features,
#                      dropout=nn.Dropout(0.1),
#                      activation=activation,
#                      norm=nn.LayerNorm(out_features),
#                      residual=residual)

def build_encoder(enc_type: str,
                  dim_model: int,
                  num_layers: int,
                  num_heads: int,
                  **enc_config):

  match enc_type:
    # case "xformers":
    #   dropout = enc_config["dropout"]
    #   max_tokens = enc_config["max_tokens"]
    #   feedward_network = enc_config["feedward_network"]
    #   activation = enc_config["activation"]
    #   xformers_config = xf.xFormerConfig(
    #     [
    #       {
    #         "block_type": "encoder",
    #         "dim_model": dim_model,
    #         "num_layers": num_layers,
    #         "residual_norm_style": "deepnorm",
    #         "multi_head_config": {
    #           "num_heads": num_heads,
    #           "residual_dropout": dropout,
    #           "use_separate_proj_weight": True,
    #           "bias": True,
    #           "attention": {
    #               "name": "scaled_dot_product",
    #               "dropout": dropout,
    #               "causal": False,
    #               "seq_len": max_tokens,
    #           },
    #           "dim_model": dim_model,
    #         },
    #         "feedforward_config": {
    #           "name": feedward_network,
    #           "dropout": dropout,
    #           "activation": activation,
    #           "hidden_layer_multiplier": 4,
    #           "dim_model": dim_model,
    #         },
    #       },
    #     ]
    #   )
    #   encoder = xf.xFormer.from_config(xformers_config)
    case "x-transformers":
      ff_mult = enc_config.get('ff_mult', 4)

      # encoder for ipr and msa
      ff_glu = enc_config.get('ff_glu', False)
      ff_swish = enc_config.get('ff_swish', False)

      pre_norm = False
      encoder = Encoder(dim = dim_model,
                        depth = num_layers,
                        heads = num_heads,
                        pre_norm = pre_norm,
                        residual_attn = True,    # add residual attention
                        ff_mult = ff_mult,
                        ff_swish = ff_swish,
                        ff_glu = ff_glu,
                       )
    
    case _:
      raise NotImplementedError(f"not implement for {enc_type}")

  return encoder

class GAUEncoder(nn.Module):
  def __init__(self, embed_dims: int,
               query_key_dim: int,
               expansion_factor: int,
               laplace_attn_fn: bool,
               depth: int, **kwargs) -> None:
    super().__init__()

    self.embed_dims = embed_dims
    self.query_key_dim = query_key_dim
    self.expansion_factor = expansion_factor
    self.laplace_attn_fn = laplace_attn_fn
    self.depth = depth

    # build the encoder
    self.encoder = \
      nn.ModuleList([nn.Sequential(
        GAU(dim = embed_dims,
            query_key_dim = query_key_dim,
            expansion_factor = expansion_factor,
            causal = False,
            laplace_attn_fn = laplace_attn_fn),
        nn.LayerNorm(embed_dims))
        for _ in range(depth)])
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

  def forward(self, x: th.Tensor):
    # encoder
    for layer in self.encoder:
      x = layer(x)
    return x


class InterProEncoder(IPR):
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
    parser = parent_parser.add_argument_group("InterPro Head")
    parser.add_argument("--msa-embed-dims", type=int, default=2048)
    parser.add_argument("--enc-embed-dims", type = int, default = 128)
    parser.add_argument("--enc-groups", type=int, default=8)
    parser.add_argument("--enc-depth", type = int, default = 6)
    parser.add_argument("--enc-num-heads", type = int, default = 8)
    parser.add_argument("--enc-pre-norm", action="store_true")
    parser.add_argument("--enc-ff-mult", type = int, default = 4)
    parser.add_argument("--enc-ff-glu", action = "store_true")
    parser.add_argument("--enc-ff-swish", action = "store_true")
    parser.add_argument("--enc-use-gau", action="store_true")
    parser.add_argument("--enc-gau-qk-dim", type=int, default=128)
    parser.add_argument("--enc-gau-mult", type=int, default=2)
    parser.add_argument("--enc-gau-laplace-attn-fn", action="store_true")
    return parent_parser

  def __init__(self, interpros: pd.DataFrame,
               embed_dims: int,
               num_classes: int,
               **kwargs,
               ) -> None:
    super().__init__()
    assert "interpros" in interpros.columns
    self.interpros = interpros
    self.mlb = prep.MultiLabelBinarizer().fit(self.interpros["interpros"])
    num_interpros = self.mlb.classes_.shape[0]
    # self.ipr_index = pd.DataFrame(list(range(num_interpros)), index=self.mlb.classes_+1) # pad is 0
    # self.interpros["interpros"] = self.interpros["interpros"].apply(lambda xs: self.ipr_index.loc[xs].to_list())
    self.ipr_index = {x: i+1 for i, x in enumerate(self.mlb.classes_)}
    # self.interpros["interpros"] = self.interpros["interpros"].apply(
    #   lambda xs: [self.ipr_index[x] for x in xs])
    self.embed_dims = embed_dims
    self.num_classes = num_classes
    self.msa_embed_dims = kwargs.get("msa_embed_dims", 2048)
    self.groups = kwargs.get("enc_groups", 8)

    # get the config of the encoder
    pre_norm = kwargs.get('enc_pre_norm', False)
    depth = kwargs.get('enc_depth', 6)
    num_heads = kwargs.get('enc_num_heads', 8)
    ff_mult = kwargs.get('enc_ff_mult', 4)
    # encoder for ipr and msa
    ff_glu = kwargs.get('enc_ff_glu', False)
    ff_glu_mult_bias = kwargs.get('enc_ff_glu_mult_bias', False)
    ff_swish = kwargs.get('enc_ff_swish', False)

    # get the config of the GAU encoder
    use_gau = kwargs.get('enc_use_gau', False)
    query_key_dim = kwargs.get('enc_gau_qk_dim', 128)
    expansion_factor = kwargs.get('enc_gau_mult', 2)
    laplace_attn_fn = kwargs.get('enc_gau_laplace_attn_fn', True)

    assert self.msa_embed_dims % self.groups == 0
    assert embed_dims % self.groups == 0
    group_embed_dims = embed_dims // self.groups

    stacked = True
    self.init_ipr_proj = nn.Sequential(
      nn.Linear(num_interpros, embed_dims),
      nn.ReLU())
    # self.init_msa_proj = GroupLinear(self.msa_embed_dims, embed_dims,
    #                                  self.groups, 
    #                                  torch_interpro_mlp,
    #                                  stacked=stacked)
    self.msa_proj = nn.Sequential(
      nn.Linear(self.msa_embed_dims // self.groups, group_embed_dims),
      nn.ReLU())

    if use_gau:
      self.encoder = GAUEncoder(embed_dims=group_embed_dims,
                                query_key_dim=query_key_dim,
                                expansion_factor=expansion_factor,
                                laplace_attn_fn=laplace_attn_fn,
                                depth=depth,
                                )
    else:
      self.encoder = Encoder(dim = group_embed_dims,
                             depth = depth,
                             heads = num_heads,
                             pre_norm = pre_norm,
                             ff_mult = ff_mult,
                             ff_swish = ff_swish,
                             ff_glu = ff_glu,
                             ff_glu_mult_bias = ff_glu_mult_bias,
                             )

    self.ln = nn.LayerNorm(group_embed_dims) if (not pre_norm and not use_gau) else nn.Identity()
    self.act = nn.ReLU()
    self.linear = nn.Linear(embed_dims * 2,
                            self.num_classes)
  
  def forward(self, x: th.Tensor):
    # x: (batch_size, embed_dims, msa_feature_h, msa_feature_w)
    # (batch_size, msa_size, embed_dims), msa_size = msa_feature_h * msa_feature_w
    x = F.adaptive_avg_pool2d(x, 1).squeeze((2,3)) # (batch_size, embed_dims)
    batch_size = x.size(0)
    if self.proteins is not None:
      assert self.proteins is not None
      ipr = th.tensor(self.mlb.transform(self.interpros.loc[self.proteins, "interpros"])).to(x.device) # type: ignore
      ipr = ipr.float() # must be float, not be int64
    else:
      # (batch_size, embed_dims)
      ipr = th.zeros(self.mlb.classes_.shape[0], device=x.device).float()
      ipr = ipr.unsqueeze(0).expand(batch_size, -1)  # no allocation of memory with expand

    ipr = self.init_ipr_proj(ipr) # (batch_size, embed_dims // 2)
    # (batch_size, self.groups, group_embed_dims // 2)
    ipr = ipr.view(batch_size, self.groups, -1)
    x = x.view(batch_size, self.groups, -1)
    x = self.msa_proj(x) # (batch_size, groups, group_embed_dims)
    # (batch_size, groups, group_embed_dims)
    x = self.ln(self.encoder(th.concat([x, ipr], dim=1)))
    return self.linear(self.act(x.flatten(1))) # embed_dims * 2 -> num_classes
  
  @classmethod
  def from_file(cls, prot_interpro_file: str,
                embed_dims: int,
                num_classes: int,
                **kwargs,):
    interpros = pd.read_pickle(prot_interpro_file)
    return cls(interpros, embed_dims, num_classes,
               **kwargs)

def ipr_encoder(prot_interpro_file: str,
                      embed_dims: int,
                      num_classes: int,
                      **kwargs):
  return InterProEncoder.from_file(
    prot_interpro_file, embed_dims, num_classes,
    **kwargs
  )

# class GroupLinear(nn.Module):
#   def __init__(self, in_dims, out_dims,
#                num_groups: int = 3,
#                linear = torch_interpro_mlp,
#                stacked: bool = False,
#                ) -> None:
#     super().__init__()
#     if out_dims < num_groups: num_groups = out_dims

#     nclasses_lst = [out_dims // num_groups] * num_groups + \
#       ([n] if (n := out_dims % num_groups) != 0 else [])
#     self.groups = nn.ModuleList(
#       [linear(in_dims, x) for x in nclasses_lst]
#     )
#     self.stacked = stacked
  
#   def forward(self, x: th.Tensor):
#     if not self.stacked:
#       return th.concat([p(x) for p in self.groups], 
#                      dim=1)
#     else:
#       return th.concat([p(x).unsqueeze(1) for p in self.groups],
#       dim=1)

# class GroupSwiGLU(nn.Module):
#   def __init__(self, in_dims, out_dims,
#                num_groups: int = 4,
#                init_linear = torch_interpro_mlp,
#                stacked: bool = False,
#                ) -> None:
#     super().__init__()
#     if out_dims < num_groups: num_groups = out_dims

#     nclasses_lst = [out_dims // num_groups] * num_groups + \
#       ([n] if (n := out_dims % num_groups) != 0 else [])
#     self.projs = nn.ModuleList(
#       [init_linear(in_dims, x) for x in nclasses_lst]
#     )
#     self.hiddens = nn.ModuleList(
#       [xsw.SwiGLU(x, x)
#        for x in nclasses_lst]
#     )
#     self.stacked = stacked
  
#   def forward(self, x: th.Tensor):
#     xs = map(lambda p: p(x), self.projs)
#     if not self.stacked:
#       return th.concat(list(map(lambda h, x: h(x), self.hiddens, xs)),
#                        dim=1)
#     else:
#       return th.concat(list(map(lambda h, x: h(x).unsqueeze(1), 
#                                 self.hiddens, xs)),
#                        dim=1)

# define head
def define_H(prot_interpro_file: str,
             embed_dims: int,
             num_classes: int, 
             head_type: str, **kwargs):
  match head_type:
    # case "head":
    #   return InterProHead.from_file(
    #     prot_interpro_file,
    #     embed_dims,
    #     num_classes,
    #     ipr_groups=16,
    #     msa_groups=None,
    #   )
    case "encoder":
      return ipr_encoder(prot_interpro_file,
                         embed_dims,
                         num_classes,
                         msa_embed_dims=2048,
                         groups=8,
                         **kwargs)
    case _:
      raise NotImplementedError(
        f"not implement head for {head_type}")
