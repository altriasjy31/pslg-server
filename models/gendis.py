## generator and discriminator
import typing as T
import functools as ft
from argparse import ArgumentParser, Namespace
import pickle
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.parallel as nn_parallel
import numpy as np

import os
import sys

prj_dir = os.path.dirname(os.path.dirname(__file__))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

from experiments.msa import MSAEncoder

import models.heads as mheads
import models.architecture as arch
from models.architecture import get_norm_layer, init_net
from models.architecture import ResnetGenerator
from resnet import resnet18, resnet34, resnet50,resnet101, resnet152
from resnet import ResNet
from resnet import resnext50_32x4d, resnext101_32x8d
from resnet import wide_resnet50_2, wide_resnet101_2
from models.timm_models import (timm_resnet, 
                         timm_resnet51q, timm_resnet61q,
                         timm_tresnet,
                         timm_nfnet,
                         timm_coat,
                         timm_caformer,
                         ml_decoder,
                         )
from models.timm_models import (TimmResNet, 
                                TimmTResNet, 
                                TimmByobNet, 
                                TimmNF,
                                TimmCoaT,
                                TimmMetaFormer,
                                MLDecoder)

from models.utils import parsing

cuda = True if torch.cuda.is_available() else False

class Arch(nn.Module):
    def __init__(self, opt : Namespace):
        """
        in_channels: int
        out_channels_G: int
        num_classes: int
        top_k: int
        max_len: int
        msa_cutoff: float
        msa_penalty: float
        msa_embedding_dim: int
        msa_encoding_strategy: str
        ngf: int
        netG: str
        no_antialias: bool
        no_antialias_up: bool
        load_gen: Optional[str]
        freeze_gen: Optional[str]
        ndf: int
        netD: str
        replace_stride_with_dilation: List[bool]
        no_dropout: bool
        normG: str
        normD: str
        init_type: str
        init_gain: float
        gpu_ids: List[int]
        no_jit: bool
        """
        super(Arch, self).__init__()

        self.in_channels = opt.in_channels
        self.out_channels_G = opt.out_channels_G
        self.num_classes = opt.num_classes
        self.gpu_ids = opt.gpu_ids

        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        need_init = True

        pre_model = MSAEncoder(opt.msa_embedding_dim, 
                                encoding_strategy=opt.msa_encoding_strategy)
        # pre_model = nn.Embedding(21, embedding_dim)

        # the weight of pre_model should also be loaded
        self.pre_model = init_net(pre_model, init_type=opt.init_type, init_gain=opt.init_gain,
                                  gpu_ids=opt.gpu_ids, initialize_weights=need_init)

        # nets = []
        # input: b * 21 * seqLen * topK
        # update
        # input: b * 441 * seqLen(patched) * seqLen(patched)
        # defaults is b * 441 * 128 * 128
        
        self.gnet = define_G(self.in_channels, self.out_channels_G,ngf=opt.ngf, netG=opt.netG,
                             norm=opt.normG, use_dropout=not opt.no_dropout,
                             init_type=opt.init_type, init_gain=opt.init_gain,
                             no_antialias=opt.no_antialias, no_antialias_up=opt.no_antialias_up,
                             gpu_ids=opt.gpu_ids,
                             need_init_weights=need_init)

        params = opt.__dict__
        ks = get_init_keys(name2model[opt.netD])
        ropt = {"params": {k: params[k] for k in ks},
                "head_params": {}}
        # ["protein_interpros","ipr_embed_dims","feature_size"]
        if opt.use_ipr_head and opt.protein_interpros is not None:
            ropt["head_params"] = {k: t # none is not selected
                                   for k in ipr_keys + ["protein_interpros"]
                                   if (t := params.get(k)) is not None}
        self.rnet = define_D(self.out_channels_G, self.num_classes, netD=opt.netD,
                             init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids,
                             **ropt)
    
    def set_proteins(self, proteins: T.List[str]):
        model = self.rnet.module if isinstance(self.rnet, nn_parallel.DataParallel) \
            else self.rnet
        if hasattr(model, "fc"):
            fc = getattr(model, "fc")
        elif hasattr(model, "head"):
            fc = getattr(model, "head")
        else:
            raise NotImplementedError("only support fc or head")
        
        assert isinstance(fc, mheads.IPR)

        fc.set_proteins(proteins)
    
    def get_proteins(self):
        model = self.rnet.module if isinstance(self.rnet, nn_parallel.DataParallel) \
            else self.rnet
        if hasattr(model, "fc"):
            fc = getattr(model, "fc")
        elif hasattr(model, "head"):
            fc = getattr(model, "head")
        else:
            raise NotImplementedError("only support fc or head")
        
        assert isinstance(fc, mheads.IPR)
        return fc.proteins
    
    def get_fc(self):
        model = self.rnet.module if isinstance(self.rnet, nn_parallel.DataParallel) \
            else self.rnet
        if hasattr(model, "fc"):
            fc = model.fc
        elif hasattr(model, "head"):
            fc = model.head
        else:
            raise NotImplementedError("only support fc or head")
        assert isinstance(fc, nn.Module)
        return fc
    
    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def freeze_all_except_fc(self):
        self.freeze_all()
        fc = self.get_fc()
        for p in fc.parameters():
            p.requires_grad = True
    
    def forward_features(self, x,
                        permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1)):
        x = self.pre_model(x)
        # x = x.permute(0, 3, 2, 1) # b c h w -> b w h c
        x = x.permute(*permute_dims)
        if self.gnet is not None:
            x = self.gnet(x)
        
        # using forward feature in rnet
        if hasattr(self.rnet, "forward_features"):
            return self.rnet.forward_features(x)
        elif hasattr(self.rnet.module, "forward_features"):
            return self.rnet.module.forward_features(x)
        else:
            # must implement the forward_feature method
            raise NotImplementedError("must implement the forward_features method")
            
    def forward(self, x : Tensor, 
                permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1)):
        x = self.pre_model(x)
        # x = x.permute(0, 3, 2, 1) # b c h w -> b w h c
        x = x.permute(*permute_dims)
        if self.gnet is not None:
            x = self.gnet(x)
        x = self.rnet(x)
        return x


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None,
             need_init_weights = True):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2.
        need_init_weights (bool) -- control whether to initialize the net weights
    """

    norm_layer = get_norm_layer(norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, 
                              n_blocks=4, opt=opt)
    elif netG == "resnet_2blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=2, opt=opt)
    elif netG == "resnet_oneblock":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up,
                              n_blocks=1, opt=opt)
    elif netG is None or netG == "none":
        net = None
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=need_init_weights) \
            if net is not None else net

name2net = {
    "resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,"resnet101": resnet101, "resnet152" : resnet152,
    "resnext50_32x4d": resnext50_32x4d, "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2":wide_resnet50_2, "wide_resnet101_2": wide_resnet101_2,
    "timm_resnet51q": timm_resnet51q,
    "timm_resnet61q": timm_resnet61q,
    "timm_resnet50d": ft.partial(timm_resnet, variant="50d"),
    "timm_ecaresnet50d": ft.partial(timm_resnet, variant="50d", eca=True),
    "timm_ecaresnet101d": ft.partial(timm_resnet, variant="101d", eca=True),
    "timm_tresnet_m": ft.partial(timm_tresnet, variant="m"),
    "timm_tresnet_l": ft.partial(timm_tresnet, variant="l"),
    "timm_tresnet_xl": ft.partial(timm_tresnet, variant="xl"),
    "timm_eca_nfnet_l0": ft.partial(timm_nfnet, nl=0),
    "timm_eca_nfnet_l1": ft.partial(timm_nfnet, nl=1),
    "timm_eca_nfnet_l2": ft.partial(timm_nfnet, nl=2),
    "timm_eca_nfnet_l3": ft.partial(timm_nfnet, nl=3),
    "timm_coat_small": ft.partial(timm_coat, type="small"),
    "timm_coat_medium": ft.partial(timm_coat, type="medium"),
    "timm_caformer_s18": ft.partial(timm_caformer, type="s18"),
    "timm_caformer_m36": ft.partial(timm_caformer, type="m36"),
    "mldecoder": ml_decoder
}

name2model = {
    "resnet18": ResNet, "resnet34": ResNet, "resnet50": ResNet,"resnet101": ResNet, "resnet152" : ResNet,
    "resnext50_32x4d": ResNet, "resnext101_32x8d": ResNet,
    "wide_resnet50_2":ResNet, "wide_resnet101_2": ResNet,
    "timm_resnet51q": TimmByobNet,
    "timm_resnet61q": TimmByobNet,
    "timm_resnet50d": TimmResNet,
    "timm_ecaresnet50d": TimmResNet,
    "timm_ecaresnet101d": TimmResNet,
    "timm_tresnet_m": TimmTResNet,
    "timm_tresnet_l": TimmTResNet,
    "timm_tresnet_xl": TimmTResNet,
    "timm_eca_nfnet_l0": TimmNF,
    "timm_eca_nfnet_l1": TimmNF,
    "timm_eca_nfnet_l2": TimmNF,
    "timm_eca_nfnet_l3": TimmNF,
    "timm_coat_small": TimmCoaT,
    "timm_coat_medium": TimmCoaT,
    "timm_caformer_s18": TimmMetaFormer,
    "timm_caformer_m36": TimmMetaFormer,
    "mldecoder": MLDecoder
}

ipr_keys = ["use_ipr_head", "ipr_embed_dims", "head_type", 
            "enc_depth",
            "enc_pre_norm",
            "enc_num_heads", 
            "enc_ff_mult",
            "enc_ff_glu", 
            "enc_ff_glu_mult_bias",
            "enc_ff_swish",
            # GAU
            "enc_use_gau",
            "enc_gau_qk_dim",
            "enc_gau_mult",
            "enc_gau_laplace_attn_fn"
            ]

def get_init_keys(net):
    if net == ResNet:
        init_keys = ["replace_stride_with_dilation", "normD"]
    elif net == MLDecoder:
        init_keys = ["mldecoder_base_model"]
    elif net == TimmCoaT:
        init_keys = ["input_size"]
    else:
        init_keys = []

    return init_keys

def define_D(in_channels : int, num_classes : int, netD : str,
             init_type='normal', init_gain=0.02, gpu_ids = [], need_init_weights = True,
             **opt):
        assert  name2net.get(netD) is not None, \
            f"{netD} is not implemented"
        
        params = opt["params"]
        norm_type = params.get("normD", None)
        if norm_type is not None:
            del params["normD"]
            params["norm_layer"] = get_norm_layer(norm_type)
        
        rnet =  name2net[netD](in_channels=in_channels, num_classes=num_classes, **params)
        # interpro_head
        head_params = opt["head_params"]
        if head_params.get("use_ipr_head", False) and \
            head_params.get("protein_interpros") is not None:
            rnet = add_head(rnet,
                            num_classes=num_classes,
                            **head_params)
        if need_init_weights:
            rnet = init_net(rnet, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

        return rnet

def add_head(model: nn.Module,
             protein_interpros: str,
             ipr_embed_dims: int,
             num_classes: int,
             head_type: str,
             **kwargs):

    if hasattr(model, 'fc'):  # resnet50, etc
        del model.fc
        model.fc = mheads.define_H(protein_interpros,
                                   ipr_embed_dims,
                                   num_classes,
                                   head_type,
                                   **kwargs)
    elif hasattr(model, 'head'):  # tresnet
        del model.head
        model.head = mheads.define_H(protein_interpros,
                                     ipr_embed_dims,
                                     num_classes,
                                     head_type,
                                     **kwargs)
    else:
        raise NotImplementedError("model is not suited for our head module")
    
    if hasattr(model, "global_pool"):
        model.global_pool = nn.Identity()
    elif hasattr(model, "avgpool"):
        model.avgpool = nn.Identity()
    else:
        pass

    return model

def main():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--in-channels", dest="in_channels", type=int, default=21)
    parser.add_argument("--out-channels-G", dest="out_channels_G", type=int, default=42)
    parser.add_argument("--num-classes",dest="num_classes", type=int, default=19939)

    parser.add_argument("--top-k",dest="top_k",type=int,default=100,
                        help="select the top k sequence ib msa")
    parser.add_argument("--max-len",dest="max_len", type=int, default=1000,
                        help="the max lenght of sequence for using")

    parser.add_argument("--msa-cutoff", dest="msa_cutoff", type=float,default=0.8,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-penalty", dest="msa_penalty", type=float,default=4.5,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-embedding-dim", dest="msa_embedding_dim", type=int, default=21,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-encoding-strategy", dest="msa_encoding_strategy", type=str, 
                        choices=["one_hot", "emb", "emb_plus_one_hot", "emb_plus_pssm","fast_dca"], 
                        default="emb_plus_one_hot",
                        help="parameters for encoding the msa file")

    parser.add_argument('--ngf', type=int, default=64, 
                        help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_4blocks', 
                        help="specify generator architecture " +
                        "[resnet_9blocks | resnet_6blocks | resnet_4blocks]")
    parser.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                        help="use dilated convolution blocks in generator")
    parser.add_argument("--no-antialias-up", dest="no_antialias_up", action="store_true",
                        help="use dilated convolution_transposed blocks in generator")
    parser.add_argument("--load-gen", dest="load_gen", type=str,
                        help="the path of pre-trained generator model")
    parser.add_argument("--freeze-gen", dest="freeze_gen", action="store_true",
                        help="control whether to freeze the generator model when training")
    parser.add_argument('--ndf', type=int, default=64, 
                            help='# of dis filters in the last conv layer')
    parser.add_argument('--netD', type=str, default='resnet50v2', 
                        help='specify discriminator architecture '+
                        '[resnet18 | resnet34 | resnet50 | resnet101 | resnet152] or ' + 
                        '[resnext50_32x4d | resnext101_32x8d]' +
                        '[wide_resnet50_2 | wide_resnet101_2]'+
                        '[basic]')

    parser.add_argument("--dilation", nargs=3, dest="replace_stride_with_dilation", 
                        default=[False, False, False], type=bool,
                        help="using dilation to replace the stride in resnet")
    parser.add_argument('--no-dropout', dest="no_dropout",action='store_true', 
                        help='no dropout for the generator', default=False)
    parser.add_argument('--normG', type=str, default='instance', 
                        help='for generator, instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--normD', type=str, default='batch', 
                            help='for discriminator, instance normalization or batch normalization [instance | batch | none]')    
     
    parser.add_argument('--init-type', dest="init_type", type=str, default='xavier', 
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init-gain', dest="init_gain", type=float, default=0.02, 
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--gpu-ids', dest="gpu_ids",type=str, default='0,1', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--no-jit", dest="no_jit", action="store_true",
                        help="not use torch.jit.script")
    opt : Namespace
    opt = parsing(parser)

    print(opt)

    model_arch = Arch(opt).cuda()
    print(model_arch)
    x = torch.randint(0, 21,(12, 40, 2000)).cuda()
    x = model_arch(x)

if __name__ == "__main__":
    main()
