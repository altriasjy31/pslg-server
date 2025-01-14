import timm.models as tmodels
import timm.layers as tlayers

from timm.models.resnet import ResNet as TimmResNet
from timm.models.tresnet import TResNet as TimmTResNet
from timm.models.byobnet import ByobNet as TimmByobNet
from timm.models.nfnet import NormFreeNet as TimmNF
from timm.models.coat import CoaT as TimmCoaT
from timm.models.metaformer import MetaFormer as TimmMetaFormer
from timm.models.convnext import ConvNeXt as TimmConvNeXt
import timm.models.tresnet as timm_tresnet

from .ml_decoder import add_ml_decoder_head, MLDecoder

def timm_resnet(in_channels, num_classes,
                variant: str = "50d",
                eca: bool = False) -> TimmResNet:
  model_name = f"resnet{variant}"
  if eca: model_name = f"eca{model_name}"
  return tmodels.create_model(model_name, 
                              in_chans=in_channels,
                              num_classes=num_classes)

def timm_resnet51q(in_channels, num_classes):
  return tmodels.create_model("resnet51q",
                              in_chans=in_channels,
                              num_classes=num_classes)

def timm_resnet61q(in_channels, num_classes):
  return tmodels.create_model("resnet61q", 
                              in_chans=in_channels,
                              num_classes=num_classes)

def timm_tresnet(in_channels, num_classes,
                 variant: str = "m") -> TimmTResNet:
  return tmodels.create_model(f"tresnet_{variant}",
                              in_chans=in_channels,
                              num_classes=num_classes)

def timm_nfnet(in_channels, num_classes,
               nl: int = 2,
               ) -> TimmNF:
  model_name = f"eca_nfnet_l{nl}"
  return tmodels.create_model(model_name,
                              in_chans=in_channels,
                              num_classes=num_classes)

def timm_coat(in_channels, num_classes,
              input_size,
              type: str = "small"):
  model_name = f"coat_lite_{type}"
  return tmodels.create_model(model_name,
                              in_chans=in_channels,
                              img_size=input_size,
                              num_classes=num_classes)

def timm_caformer(in_channels, num_classes,
                  type: str = "m36"):
  model_name = f"caformer_{type}"
  return tmodels.create_model(model_name,
                              in_chans=in_channels,
                              num_classes=num_classes)

def timm_convnext(in_channels, num_classes,
                  type: str = "small"):
  model_name = f"convnext_{type}"
  return tmodels.create_model(model_name,
                              in_chans=in_channels,
                              num_classes=num_classes)

def ml_decoder(in_channels, num_classes,
               mldecoder_base_model: str = "tresnet_xl"):
  model = tmodels.create_model(mldecoder_base_model,
                               in_chans=in_channels,
                               num_classes=num_classes)
  return add_ml_decoder_head(model)