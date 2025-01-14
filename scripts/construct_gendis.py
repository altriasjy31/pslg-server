import sys
import os

prj_dir = os.path.dirname(os.path.dirname(__file__))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

import typing as T
from models.utils import parsing
from experiments.go.train import training_prog
from experiments.go.train_v2 import training_prog as training_v2
from experiments.go.train_ipr import training_prog as ipr_training
from experiments.go.eval_ipr import evaluation_prog as ipr_eval
from experiments.go.eval import evaluation_prog

from experiments.go.train_im import training_prog as im_training
from experiments.go.eval_im import evaluation_prog as im_eval
from experiments.preprocess.utils import get_surfix
from argparse import ArgumentParser, Namespace

# from models.im import IMEncoder
from models.heads import InterProEncoder

def main():
    """
    # Dataset Parameters
    dataset_path: str,
    working_dir: str,
    n_classes: int,
    mode: str, # [train, test]
    task: str, # [cc, mf, bp]
    file_type:str, # file surfix like npy
    top_k: int, # the number of top sequence in msa for selection
    max_len: int, # the max length of sequence

    # MSAModel Parameters
    output_nc, # the number of output channels
    ngf: int, # the number of encoder last layer filters
    netG: str, # the type of encoder name
    n_layers: int, # the numger of cnn decoder layers
    no_dropout: bool,
    norm: str, # the norm type
    init_type: str,
    init_gain: float,
    gpu_ids: List[int],

    # Training Parameters
    epochs: int,
    lr: float,
    batch_size: int,
    validation_split: float,
    shuffle_dataset: bool
    model_saving: str
    """
    parser = ArgumentParser(description='GO prediction: GenDis model construction')

    # positional arguments
    parser.add_argument("file_address", metavar="dataset_path")
    parser.add_argument("working_dir", help="the msa data directory")
    parser.add_argument("model_saving", help="the path for model saving")

    # config filepath
    parser.add_argument("-c", "--config", type=str,
                        help="config filepath")

    # optional arguments
    parser.add_argument("--in-channels", dest="in_channels", type=int, default=21)
    parser.add_argument("--out-channels-G", dest="out_channels_G", type=int, default=21)
    parser.add_argument("--num-classes",dest="num_classes", type=int, default=19939)
    parser.add_argument("--mode", choices=["train","test", 
                                           "train_ipr", "test_ipr",
                                           "train_im", "test_im"], default="train")
    parser.add_argument("--task",choices=["cellular_component",
                                          "molecular_function",
                                          "biological_process"], default="biological_process")
    parser.add_argument("--top-k",dest="top_k",type=int,default=40,
                        help="select the top k sequence ib msa")
    parser.add_argument("--max-len",dest="max_len", type=int, default=2000,
                        help="the max lenght of sequence for using")
    parser.add_argument("--dataloader-num-workers", dest="dataloader_num_workers", type=int, default=5,
                        help="the number of workers that dataloader will need")

    parser.add_argument("--msa-embedding-dim", dest="msa_embedding_dim", type=int, default=21,
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-encoding-strategy", dest="msa_encoding_strategy", type=str, 
                        choices=["one_hot", "emb", "emb_plus_one_hot", "emb_plus_pssm","fast_dca"], 
                        default="emb_plus_one_hot",
                        help="parameters for encoding the msa file")
    parser.add_argument("--msa-max-size", dest="msa_max_size", type=int, default=10000,
                        help="maximum alignments when read msa file")
    parser.add_argument('--ngf', type=int, default=64, 
                        help='# of gen filters in the last conv layer')
    parser.add_argument('--netG', type=str, default='resnet_9blocks', 
                        help="specify generator architecture " +\
                        "[resnet_9blocks | resnet_6blocks | resnet_4blocks | " +
                        "resnet_2blocks | resnet_oneblock | none]")
    parser.add_argument("--no-antialias", dest="no_antialias", action="store_true",
                        help="use dilated convolution blocks in generator")
    parser.add_argument("--no-antialias-up", dest="no_antialias_up", action="store_true",
                        help="use dilated convolution_transposed blocks in generator")
    parser.add_argument('--ndf', type=int, default=64, 
                            help='# of dis filters in the last conv layer')
    parser.add_argument('--netD', type=str, default='resnet50', 
                        help='specify discriminator architecture '+
                        '[resnet ... such as renset50]' + 
                        '[timm_tresnet_m, ...]' +
                        '[eca_nfnet_l0, l1, l2]')
    parser.add_argument("--dilation", nargs=3, dest="replace_stride_with_dilation", 
                        default=[False, False, False], type=bool,
                        help="using dilation to replace the stride in resnet")

    parser.add_argument('--no-dropout', dest="no_dropout",action='store_true', 
                        help='no dropout for the generator')
    parser.add_argument('--normG', type=str, default='instance',
                        help='for generator, instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--normD', type=str, default='batch', 
                            help='for discriminator, instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument("--input-size", nargs=2, default=[2000,40], type=int)    
    parser.add_argument("--permute-dims", nargs=4, default=[0,3,2,1], type=int)

    # use InterPro Head in discriminator
    parser.add_argument("--use-ipr-head", action="store_true",
                        help="use interpro head in netD, instead of the original")

    # InterPro Head
    parser.add_argument("--protein-interpros", type=str,
                        help="interpros dataframe path, " + 
                        "replace the fc or head to the InterProHead")
    parser.add_argument("--num-interpros", type=int)
    parser.add_argument("--no-ipr-input", action="store_true")
    parser.add_argument("--ipr-embed-dims", type=int, default=2048)
    parser.add_argument("--head-type", type=str, choices={"head", "transformer"},
                        default="head",
                        help="using interpro classifier {head, transformer} to replace the original one")

    # pretrained option
    parser.add_argument("--ipr-pretrained", type=str,
                        help="the path of pretrained ipr model (IPRFormer)")
    parser.add_argument("--msa-pretrained", type=str,
                        help="the path of pretrained msa model (Arch)")
    
    # parser = IMEncoder.add_model_specific_args(parser)
    parser = InterProEncoder.add_model_specific_args(parser)
    # Specify the hyperparams of this model
    # enc_embed_dims: the embedding dimension of encoder
    # enc_depth: the depth of encoder
    # enc_num_heads: the number of heads in encoder
    # enc_ff_mult: the multiplier of feedforward in encoder
    # enc_ff_glu: whether to use GLU in feedforward of encoder
    # enc_ff_swish: whether to use swish in feedforward of encoder

    # other parameters for training or evaluating
    parser.add_argument('--init-type', dest="init_type", type=str, default='normal', 
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init-gain', dest="init_gain", type=float, default=0.02, 
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--gpu-ids', dest="gpu_ids",type=str, default='0,1', 
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser.add_argument("--loss", type=str, default="ASL", choices=["CE", "Focal", "ASL"],
                        help="the loss function for training [CE | Focal | ASL]")
    choosing_optims = ["Ranger21", "Adam", "SGD", "AMSGrad", "RAdam", "AdaBelief"]
    parser.add_argument("--optim", type=str, default="Adam", choices=choosing_optims,
                        help=f"select the optimizer {' | '.join(choosing_optims)}")
    parser.add_argument("--weight-decay", type=float,
                        help="set the weight decay to optimizer")
    parser.add_argument("--no-model-weight-decay", action="store_true",
                        help="using the weight_decay of optim instead of using 'add_weight_decay' for model")

    parser.add_argument("--epochs",type=int, default=100)
    parser.add_argument("--stop-epoch", dest="stop_epoch", default=60, type=int)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--lr-policy", dest="lr_policy", default='cycle', 
                        help='learning rate policy. [cycle | multi]')
    parser.add_argument("--pct-start", default=0.2,type=float,
                        help="The percentage of the cycle (in number of steps) spent increasing the learning rate.")
    parser.add_argument("--milestones", nargs="+", type=int, default=[5,20],
                        help="List of epoch indices. Must be increasing. Multisteplr")
    parser.add_argument("--multistep-gamma", type=float, default=0.1,
                        help="gamma for multstep lr")
    parser.add_argument("--base-lr", type=float,
                        help="min lr for CyclicLR")

    parser.add_argument('--batch-size', dest="batch_size",default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument("--validation-split", dest="validation_split", type=float, default=0.0,
                        help="the ratio of size about validation dataset  and original dataset")
    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")

    parser.add_argument("--load",dest="trained_model",
                        help="the model state dict filename")
    parser.add_argument("--no-eval", dest="no_eval", action="store_true",
                        help="not evaluate the result")
    parser.add_argument("--prediction-save", dest="prediction_save",
                        help="the file address for saving prediction result")
    parser.add_argument("--v2", action="store_true")

    prog_name, _ = get_surfix(os.path.basename(__file__))
    parser.add_argument("--for-retrain", dest="for_retrain", type=str,
                        help="the model state dict for retrain")
    parser.add_argument("--full-fine-tuning", action="store_true",
                        help="not freeze")

    parser.add_argument('--wandb', type=str, default=f'som-{prog_name}', 
            help="wandb project name")
    
    opt = parsing(parser)

    mode = opt.mode

    # print(opt)
    match mode:
        case "train":
            assert hasattr(opt, "optim")
            if opt.v2:
                training_v2(opt)
            else:
                training_prog(opt)
        case "test":
            evaluation_prog(opt)
        
        case "train_ipr":
            ipr_training(opt)
        
        case "test_ipr":
            ipr_eval(opt)

        case "train_im":
            im_training(opt)
        
        case "test_im":
            im_eval(opt)
        
        case _:
            raise NotImplementedError(f"the mode of {mode} is not implemented")

if __name__ == "__main__":
    main()