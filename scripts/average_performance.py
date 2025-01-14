import sys
import os

prj_dir = os.path.dirname(os.path.dirname(__file__))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

import helper_functions.metrics as hm
import experiments.msa as D
import models.gendis as G
import helper_functions.helpers as helper
from models.utils import parsing

import typing as T
import numpy as np
import argparse as P
import tqdm
import torch
import torch as th
import torch.utils.data as ud
import torch.cuda.amp.autocast_mode as am

def eval(opt: P.Namespace):
  """
  """
  torch.multiprocessing.set_sharing_strategy('file_system')
  torch.manual_seed(3407) # manual_seed 3407 is all your neeed
  batch_size = opt.batch_size

  # dataset = MSADataset(opt.file_address, opt.working_dir, opt.num_classes, opt.mode,
  #                          opt.task, top_k = opt.top_k, max_len = opt.max_len)
  dataset = D.MSADataset(opt.file_address, opt.working_dir, opt.mode, opt.task,
      opt.num_classes, opt.top_k, opt.max_len,
      need_proteins=not opt.no_ipr_input)

  model = G.Arch(opt)

  if opt.torch_compile:
    assert hasattr(torch, "compile"), "need pytorch > 2.0"
    model = torch.compile(model)
 
  eval_loader = ud.DataLoader(dataset, batch_size=batch_size,
                           shuffle=False,num_workers=opt.dataloader_num_workers)
  model_dir = opt.model_saving
  model_name = opt.trained_model
  state_path = os.path.join(model_dir, model_name)
  state_dict = torch.load(state_path)
  pred_savingpath = opt.prediction_save
  model.load_state_dict(state_dict=state_dict)
  model.eval()

  Sig = torch.nn.Sigmoid()
  preds_regular = []
  targs= []
  for j in range(opt.num_samplings):
    print(f"sampling: {j}")
    for i, (X, y) in tqdm.tqdm(enumerate(eval_loader)):
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
        with am.autocast():
          output_regular : th.Tensor = Sig(model(X)).cpu()
      if j == 0:
        preds_regular.append(output_regular.detach())
        targs.append(y.detach())
      else:
        preds_regular[i] += output_regular.detach()
      model.zero_grad()
  target_tensor, pred_tensor = torch.cat(targs), torch.cat(preds_regular)
  pred_tensor /= opt.num_samplings # average
  Fmax = hm.fmax_torch(target_tensor.cpu(), pred_tensor.cpu())
  AuPRC = hm.AuPRC_torch(target_tensor.cpu(), pred_tensor.cpu())
  print(f"Fmax score {Fmax:.4f}%, AuPRC socre {AuPRC:.4f}%")

  if pred_savingpath is not None:
    np.save(pred_savingpath, th.stack([target_tensor.cpu(), pred_tensor.cpu()],dim=0).cpu().numpy())


def main():
  parser = P.ArgumentParser()
  # positional arguments
  parser.add_argument("file_address", metavar="dataset_path")
  parser.add_argument("working_dir", help="the msa data directory")
  parser.add_argument("model_saving", help="the path for model saving")
  # config filepath
  parser.add_argument("-c", "--config", type=str,
                      help="config filepath")
  parser.add_argument("-n", "--num-samplings", type=int, default=5)
  parser.add_argument("--load", dest="trained_model", type=str)
  parser.add_argument("-ps", "--prediction-save", type=str)
  
  opt = parsing(parser)
  eval(opt)

if __name__ == "__main__":
  main()