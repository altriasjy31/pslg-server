import typing as T
from argparse import Namespace
import json
from time import strftime, localtime
import pickle
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch.multiprocessing as mp

import helper_functions.metrics

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

import torch.cuda.amp as amp
import torch.cuda.amp.grad_scaler as grad_scaler
import torch.cuda.amp.autocast_mode as autocast_mode

import torch.nn.utils as nn_utils
import torch.utils.data as utils_data
import torch.nn.parallel as nn_parallel
from torch.optim import lr_scheduler, optimizer
from functools import reduce
import functools as ft
import itertools as it

from models import Arch
from models.utils import to_np
import helper_functions.helpers as helper
from loss_functions.loss import AsymmetricLoss, AsymmetricLossOptimized

from loss_functions.loss import FocalLossV2 as FocalLoss
import optimizers.optim as myoptim
import timm.optim as top


# from experiments.go.GO import MSADataset
import experiments.msa as D

import wandb

Tensor = torch.Tensor

name2loss = dict(
    CE=BCEWithLogitsLoss,
    Focal=FocalLoss,
    # ASL=AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    ASL=ft.partial(AsymmetricLossOptimized, 
                   gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
)

name2optim = dict(
    Adam=ft.partial(torch.optim.Adam, eps=1e-4),
    SGD=ft.partial(torch.optim.SGD, momentum=0.9),
    AMSGrad=ft.partial(torch.optim.Adam, amsgrad=True,
                       eps=1e-4,),
    RAdam=torch.optim.RAdam,
    EAdam=ft.partial(myoptim.EAdam, amsgrad=True),
    AdaBelief=ft.partial(top.create_optimizer_v2, opt="adabelief",
                         ),
    Lion=ft.partial(top.create_optimizer_v2, opt="lion")
)

def get_scheduler(n: str, 
                  optim,
                  **kwargs):
  match n:
    case "cycle":
      assert kwargs.get("max_lr") is not None
      assert kwargs.get("steps_per_epoch") is not None
      assert kwargs.get("pct_start") is not None
      assert kwargs.get("epochs") is not None
      return lr_scheduler.OneCycleLR(optim, 
                                     max_lr=kwargs["max_lr"],
                                     steps_per_epoch=kwargs["steps_per_epoch"],
                                     epochs=kwargs["epochs"],
                                     pct_start=kwargs["pct_start"])
    case "multi":
      assert kwargs.get("milestones") is not None
      assert kwargs.get("gamma") is not None
      return lr_scheduler.MultiStepLR(optim,
                                      milestones=kwargs["milestones"],
                                      gamma=kwargs["gamma"])
    case "cyclic":
      assert kwargs.get("base_lr") is not None
      assert kwargs.get("max_lr") is not None
      base_lr = kwargs["base_lr"]
      max_lr = kwargs["max_lr"]
      assert isinstance(base_lr, float) or isinstance(base_lr, list)
      assert isinstance(max_lr, float) or isinstance(max_lr, list)
      if kwargs.get("mode") is not None:
        mode = kwargs["mode"]
        assert isinstance(mode, str)
      else:
        mode = "triangular"
      return lr_scheduler.CyclicLR(optim,
                                   base_lr=base_lr,
                                   max_lr=max_lr,
                                   mode=mode,
                                   cycle_momentum=False)
      
    case _:
      raise NotImplementedError(f"no impelement for {n}")

def validate_multi(val_loader, model, ema_model,
                   permute_dims: T.Tuple[int, int, int, int] = (0,3,2,1)):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targs = []
    for _, (X, y) in enumerate(val_loader):
        if isinstance(X, torch.Tensor):
            X = X.cuda()
        else:
            assert isinstance(X, T.List)
            proteins, X = X
            assert isinstance(X, torch.Tensor)
            X = X.cuda()
            model.set_proteins(proteins)
            ema_model.module.set_proteins(proteins)
        y = y.float().cuda()
        # compute output
        with torch.no_grad():
            with autocast_mode.autocast():
                output_regular = Sig(
                    model(X, permute_dims=permute_dims)
                )
                output_ema = Sig(
                    ema_model.module(X, permute_dims=permute_dims)
                )

        # for fmax and AuPRC calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targs.append(y.cpu().detach())

    Y, reg, ema = torch.cat(targs), \
                  torch.cat(preds_regular), \
                  torch.cat(preds_ema)
    report_reg = helper_functions.metrics.evalperf_torch(Y, reg, threshold=True, auprc=True,
                                                         no_zero_classes=True)
    report_ema = helper_functions.metrics.evalperf_torch(Y, ema, threshold=True, auprc=True,
                                                         no_zero_classes=True)
    f1 = report_reg["fmax"]
    a1 = report_reg["auprc"]
    f2 = report_ema["fmax"]
    a2 = report_ema["auprc"]
    print("fmax score regular {:.8f}, fmax score EMA {:.8f}".format(f1, f2))
    print("AuPRC score regular {:.8f}, AuPRC score EMA {:.8f}".format(a1, a2))
    wandb.log({"fmax score regular": f1, "fmax score EMA": f2})
    wandb.log({"AuPRC score regular": a1, "AuPRC score EMA": a2})
    return f2 if f2 > f1 else f1, \
    a2 if a2 > a1 else a1


def training_prog(opt: Namespace):
    """
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    torch.manual_seed(3407) # manual_seed 3407 is all your need

    # dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.num_classes, opt.mode,
    #                      opt.task, top_k=opt.top_k, max_len=opt.max_len,
    #                      msa_buffer_size=opt.msa_buffer_size)
    dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
        opt.num_classes, opt.top_k, opt.max_len, 
        need_proteins=not opt.no_ipr_input,
        msa_max_size=opt.msa_max_size,
        )

    model = Arch(opt)

    print(model)
    print(opt)

    if opt.for_retrain is not None:
        trained_state_dict = torch.load(opt.for_retrain)
        trained_state_dict = {(k.removeprefix("_orig_mod.")
                               if "_orig_mod." in k else k): v 
                               for k, v in trained_state_dict.items()}
        unloaded_state_dict = model.state_dict()
        for k, v in trained_state_dict.items():
            if k in unloaded_state_dict:
                unloaded_state_dict[k] = v
        model.load_state_dict(unloaded_state_dict)
        if not opt.full_fine_tuning:
            model.freeze_all_except_fc()

        model.train()
        print("Re-train")

    timestamp = strftime("%y%m%d%H%M%S", localtime())
    option_path = os.path.join(opt.model_saving,
                               "training_option_{}.pkl".format(timestamp))
    with open(option_path, "wb") as h:
        pickle.dump(opt.__dict__, h)
    
    # save as json
    option_path = option_path.removesuffix(".pkl") + ".json"
    with open(option_path, "w") as h:
        json.dump(opt.__dict__, h)

    # # only torch 2.0
    if opt.torch_compile:
      assert hasattr(torch, "compile"), "need pytorch > 2.0"
      model = torch.compile(model)
    ema = helper.ModelEma(model)

    lr = opt.lr
    Epochs = opt.epochs
    batch_size = opt.batch_size
    validation_split = opt.validation_split
    shuffle_dataset = opt.shuffle
    optim_name = opt.optim

    if validation_split != 0:
        random_seed = 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = utils_data.SubsetRandomSampler(train_indices)
        valid_sampler = utils_data.SubsetRandomSampler(val_indices)

        # shuffle is not compatible with sampler
        train_loader = utils_data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=opt.dataloader_num_workers)

        val_loader = utils_data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=valid_sampler,
                                                 num_workers=opt.dataloader_num_workers)
    elif opt.extra_val is not None:
        assert isinstance(opt.extra_val, str)
        # extra_val == "valid"
        val_dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.extra_val, opt.task,
                                      opt.num_classes, opt.top_k, opt.max_len,
                                      need_proteins=not opt.no_ipr_input)
        train_loader = utils_data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=opt.dataloader_num_workers,
                                             shuffle=shuffle_dataset)
        val_loader = utils_data.DataLoader(val_dataset, batch_size=batch_size,
                                           num_workers=opt.dataloader_num_workers,
                                           shuffle=shuffle_dataset)
    else:
        train_loader = utils_data.DataLoader(dataset, batch_size=batch_size,
                                                   num_workers=opt.dataloader_num_workers,
                                                   shuffle=shuffle_dataset)
        val_loader = None

    # set optimizer
    # weight_decay = 1e-4
    # weight_decay = 3e-6
    # weight_decay = 2.5e-2
    weight_decay = opt.weight_decay
    steps_per_epoch = len(train_loader)
    # loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    loss_func = name2loss[getattr(opt, "loss", "ASL")]()
    parameters = helper.add_weight_decay(model, weight_decay) \
        if not getattr(opt, "no_model_weight_decay", False) \
            else model.parameters()
    # optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    optimizer = name2optim[optim_name](parameters, lr=lr, 
                                       weight_decay=(weight_decay 
                                                     if getattr(opt, "no_model_weight_decay", False) 
                                                     else 0))

    # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,  # type: ignore
    #                                     pct_start=opt.pct_start) # pct_start = 0.2 for 100 epoch
    match opt.lr_policy:
      case "cycle":
        scheduler = get_scheduler(opt.lr_policy,
                                  optimizer,
                                  max_lr=lr,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=Epochs,
                                  pct_start=opt.pct_start)
      case "multi":
        milestones = [x * steps_per_epoch for x in opt.milestones]
        gamma = float(getattr(opt, "multistep_gamma", 0.1))
        scheduler = get_scheduler(opt.lr_policy,
                                  optimizer,
                                  milestones=milestones,
                                  gamma=gamma)
    
      case "cyclic":
        scheduler = get_scheduler(opt.lr_policy,
                                  optimizer,
                                  base_lr=opt.base_lr,
                                  max_lr=lr,
                                  #mode=opt.cyclic_mode
                                  )
      case _:
        raise NotImplementedError(f"no implement for {opt.lr_policy}")

    highest_fmax, highest_AuPRC = 0, 0
    highest_result = (0,0)
    scaler = grad_scaler.GradScaler()


    timestamp = strftime("%y%m%d%H%M%S", localtime())
    saving_info_path = os.path.join(opt.model_saving, f"training_info_{timestamp}.pkl")

    wandb_name = f"{opt.netG}-{opt.netD}-{opt.task}"

    wandb.init(project=f'{opt.wandb}', name=f'{wandb_name}')

    trainInfoList = []
    print_interval = 100
    log_interval = 10
    permute_dims = getattr(opt, "permute_dims", (0,3,2,1))
    for epoch in range(1, Epochs+1):
        if epoch > opt.stop_epoch:
            break
        
        for i, (inputData, target) in enumerate(train_loader):
            if isinstance(inputData, torch.Tensor):
                inputData = inputData.cuda()
            else:
                assert isinstance(inputData, T.List)
                proteins, inputData = inputData
                assert isinstance(inputData, torch.Tensor)
                inputData = inputData.cuda()
                model.set_proteins(proteins)
                ema.module.set_proteins(proteins)
            target = target.float().cuda()  # (batch,num_classes)

            with autocast_mode.autocast():  # mixed precision
                output = model(inputData, permute_dims=permute_dims)  # sigmoid will be done in loss !

            # print(torch.any(torch.isnan(output)))
            # assert not torch.any(torch.isnan(output)), "The output exists nan"
            loss = loss_func(output, target) # type: ignore
            model.zero_grad()

            # using scaler instead of theirselves

            scale_loss = scaler.scale(loss)
            assert isinstance(scale_loss, Tensor)

            scale_loss.backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            ema.update(model)

            # store information
            if i % print_interval == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.4e}, Loss: {:.8f}'
                    .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                            scheduler.get_last_lr()[0], \
                            loss.item()))

            if i % log_interval == 0:
                wandb.log({"Learing Rate": scheduler.get_last_lr()[0],
                        "Loss": to_np(loss)})
        if val_loader is not None:
            model.eval()
            fmax_score, AuPRC_score = validate_multi(val_loader, model, ema, 
                                                     permute_dims=permute_dims)
            model.train()
            if fmax_score > highest_result[0] and AuPRC_score > highest_result[1]:
                highest_result = fmax_score, AuPRC_score
                save_state_dict(model,
                os.path.join(opt.model_saving, 'model-highest.pth'))
                save_state_dict(ema.module,
                                os.path.join(opt.model_saving, 'ema-model-highest.pth'))
            if fmax_score > highest_fmax:
                highest_fmax = fmax_score
                save_state_dict(model,
                os.path.join(opt.model_saving, 'fmax-highest.pth'))
                save_state_dict(ema.module,
                                os.path.join(opt.model_saving, 'ema-fmax-highest.pth'))
            if AuPRC_score > highest_AuPRC:
                highest_AuPRC = AuPRC_score
                save_state_dict(model,
                os.path.join(opt.model_saving, 'AuPRC-highest.pth'))
                save_state_dict(ema.module,
                                os.path.join(opt.model_saving, 'ema-AuPRC-highest.pth'))
            print('highest_result = ({:.8f}, {:.8f})'.format(*highest_result))
            print('current_fmax = {:.8f}, highest_fmax = {:.8f}'.format(fmax_score, highest_fmax))
            print('current_AuPRC = {:.8f}, highest_AuPRC = {:.8f}\n'.format(AuPRC_score, highest_AuPRC))
        
        # save current model
        save_state_dict(model,
        os.path.join(opt.model_saving, "model-last.pth"))

    with open(saving_info_path, "wb") as h:
        pickle.dump(trainInfoList, h)

def save_state_dict(model: T.Union[Arch, nn_parallel.DataParallel], saving_path: str):
    torch.save(model.module.state_dict()
    if isinstance(model, nn_parallel.DataParallel)
    else model.state_dict(),
    saving_path)

def freeze_msa_encoder(model: Arch):
  """freeze the msa encoder and gnet layers in architecture"""
  assert isinstance(model, Arch)
  assert isinstance(model.pre_model, nn.Module)
  assert isinstance(model.gnet, nn.Module)

  for param in model.pre_model.parameters():
      param.requires_grad = False

  for param in model.gnet.parameters():
      param.requires_grad = False
  print("-------------------------------------------------------------")
  print("#############################################################")
  print("freezed the pre_model and gnet")
  print("#############################################################")
  print("-------------------------------------------------------------")