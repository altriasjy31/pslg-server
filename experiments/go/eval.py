from argparse import Namespace
import typing as T
from torch import Tensor

import helper_functions.metrics
from models import Arch
import sys
import os

from torch.cuda.amp.autocast_mode import autocast
amp_flag = True

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

# from models import MSAModel
# from models import Coevo
# from experiments.go.GO import MSADataset
import experiments.msa as D

import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import reduce
import helper_functions.helpers as helper

import tqdm

def evaluation_prog(opt : Namespace):
    """
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(3407) # manual_seed 3407 is all your neeed

    batch_size = opt.batch_size

    # dataset = MSADataset(opt.file_address, opt.working_dir, opt.num_classes, opt.mode,
    #                          opt.task, top_k = opt.top_k, max_len = opt.max_len)
    dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
        opt.num_classes, opt.top_k, opt.max_len,
        need_proteins=not opt.no_ipr_input,
        msa_max_size=opt.msa_max_size,)

    model = Arch(opt)

    if opt.torch_compile:
      assert hasattr(torch, "compile"), "need pytorch > 2.0"
      model = torch.compile(model)

    eval_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=opt.shuffle,num_workers=opt.dataloader_num_workers)
    datasize = len(dataset)
    model_dir = opt.model_saving
    model_name = opt.trained_model
    Epochs = opt.epochs
    state_path = os.path.join(model_dir, model_name)
    state_dict = torch.load(state_path)
    no_eval = opt.no_eval
    pred_savingpath = opt.prediction_save
    permute_dims = getattr(opt, "permute_dims", (0,3,2,1))
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    fmax_score_regular, auprc_score_regular, tp_tensor = eval_multi(eval_loader, model, no_eval,
                                                                    permute_dims)

    outfmt ="fmax score regular {:.4f}%, " + \
            "AuPRC score regular {:.4f}%"
    print(outfmt.format(fmax_score_regular, auprc_score_regular))

    if pred_savingpath is not None:
        np.save(pred_savingpath, tp_tensor.cpu().numpy())

def eval_multi(dataloader : DataLoader, model, no_eval: bool,
               permute_dims: T.Tuple[int, int, int, int] = (0, 3, 2, 1)):
    print("starting evaluation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targs= []

    for i, (X, y) in tqdm.tqdm(enumerate(dataloader)):
        if isinstance(X, torch.Tensor):
            X = X.cuda()
        else:
            assert isinstance(X, T.List)
            proteins, X = X
            assert isinstance(X, torch.Tensor)
            X = X.cuda()
            model.set_proteins(proteins)
        y = y.cuda()
        # print(input.shape)
        # compute output
        # print(torch.isnan(input))

        with torch.no_grad():
            with autocast():
                output_regular : Tensor = Sig(
                    model(X, permute_dims=permute_dims)
                )


        preds_regular.append(output_regular.cpu().detach())
        targs.append(y.cpu().detach())

        model.zero_grad()
    
    y_true, y_score = torch.cat(targs), \
        torch.cat(preds_regular)
    if not no_eval:
        report = helper_functions.metrics.evalperf_torch(y_true, y_score, 
                                                         threshold=True,
                                                         auprc=True,
                                                         no_zero_classes=True)
        f1 = report["fmax"]
        a1 = report["auprc"]
    else:
        f1, a1 = -1, -1

    # print(pr_mat)
    return f1, a1, torch.stack([y_true.cpu(), y_score.cpu()],dim=0)