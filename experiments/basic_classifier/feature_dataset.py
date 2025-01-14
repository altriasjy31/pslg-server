import typing as T
import pathlib
import sys
import os
prj_dir = str(pathlib.Path(__file__).parent.parent.parent)
if prj_dir not in sys.path:
    sys.path.append(prj_dir)

import experiments.dataloader as DL

import pickle
import json
import yaml
import re
import argparse as P

import torch
import torch as th
import torch.utils.data as ud

import numpy as np

SUFFIX=re.compile(r"\.([^.]+)$")
def get_suffix(s: str):
  matched = SUFFIX.search(s)
  if matched is not None:
    return matched.group(1)
  else:
    return ""

class FeatureDataset(ud.Dataset):
  namekey = "proteins"
  labelkey = "prop_annotations"
  num_classes=40000
  classes=th.zeros(num_classes, dtype=th.int)
  feature_format = "npy"
  need_proteins = False
  def __init__(self, dataset: T.Dict[str, T.Dict[str, T.Dict[str, T.List]]],
               feature_dir: str,
               mode: str, # {train, test}
               task: str, # {cco, mfo, bpo},
               num_classes: int,
               feature_dims: int,
               need_proteins: bool = False,
               ) -> None:
    super().__init__()
    
    assert mode in ["train", "test"], \
     f"only train or test, instead of {mode}"
    assert task in ["cellular_component", 
                    "molecular_function", 
                    "biological_process"], \
     f"only cco, mfo, or bpo, instead of {task}"
    self.num_classes = num_classes
    self.feature_dims = feature_dims
    self.need_proteins = need_proteins

    self.targets = self._load_data(dataset, mode, task, feature_dir,
                                   self.feature_format)
    assert "feature" in self.targets, "unexpected errors !, check the source code"

    self.len = len(self.targets["feature"])
    print(f"Loaded {mode}-set, length: {self.len}")
  
  def __len__(self):
     return self.len

  def _load_data(self,
                data: T.Dict[str, T.Dict[str, T.Dict[str, T.List]]],
                mode: str,
                task: str,
                feature_dir: str,
                feature_format: str = "npy"):
    """
    """

    subdata = data[mode][task]
    subdata["feature"] = \
      [p for x in subdata[self.namekey]
       if os.path.exists(p := os.path.join(feature_dir, f"{x}.{feature_format}"))]
    return subdata

  def get(self, key, index):
    return self.targets[key][index]
  
  def __getitem__(self, index):
    protein = self.get(self.namekey, index)
    featpath = self.get("feature", index)
    X = th.tensor(np.load(featpath))

    assert (d := X.size(0)) == self.feature_dims, \
    f"the recived feature dims of X not equal to {self.feature_dims} " + \
    f"which is {d}"

    y = self.get(self.labelkey,index)
    assert isinstance(y, T.List)

    self.classes.fill_(0)
    self.classes[y] = 1

    if self.need_proteins:
      return protein, (X, self.classes[:self.num_classes].clone())
    else:
      return X, self.classes[:self.num_classes].clone()
  
  @classmethod
  def from_file(cls, datasetpath: str,
                feature_dir: str,
                mode: str,
                task: str,
                num_classes: int,
                feature_dims: int,
                **kwargs,
                ):
    match get_suffix(datasetpath):
      case "pt":
        dataset = th.load(datasetpath)
      case "pkl":
        with open(datasetpath, "rb") as h:
          dataset = pickle.load(h)
      case "json":
        with open(datasetpath, "r") as h:
          dataset = json.load(h)
      case "yml":
        with open(datasetpath, "r") as h:
          dataset = yaml.safe_load(h)
      case _:
        raise NotImplementedError("only support pt, pkl, json, yml" +
                                  f"not support {datasetpath}")
    return cls(dataset, feature_dir,
               mode, task, num_classes, 
               feature_dims)

@DL.get_dataloader
def feature_dataloader(dataset,
                       feature_dir,
                       mode,
                       task,
                       num_classes,
                       feature_dims,
                       **datasetconfig,
                       ):
  dataset = FeatureDataset.from_file(dataset,
                                     feature_dir,
                                     mode,task,
                                     num_classes, feature_dims,
                                     **datasetconfig)
  return dataset

def test_dataloader():
  parser = P.ArgumentParser()
  parser.add_argument("dataset")
  parser.add_argument("feature")
  parser.add_argument("-m", "--mode", required=True,
                      choices=["train", "test"])
  parser.add_argument("-t", "--task", required=True,
                      choices=["cellular_component", 
                               "molecular_function", 
                               "biological_process"])
  parser.add_argument("-N", "--num-classes", type=int,
                      required=True)
  parser.add_argument("-fd", "--feature-dims", type=int,
                      default=2048)
  parser.add_argument("-bs", "--batch-size", type=int,
                      default=32)
  parser.add_argument("-ndw", "--num-dataloader-workers", type=int,
                      default=5)
  parser.add_argument("--not-shuffle", action="store_true")

  opt = parser.parse_args()

  dataloader = feature_dataloader(dataset=opt.dataset,
                                  feature_dir=opt.feature,
                                  mode=opt.mode,
                                  task=opt.task,
                                  num_classes=opt.num_classes,
                                  feature_dims=opt.feature_dims,
                                  batch_size=opt.batch_size,
                                  dataloader_num_workers = opt.num_dataloader_workers,
                                  shuffle_dataset = not opt.not_shuffle,
                                  validation_split=0.,)
  assert not isinstance(dataloader, tuple)
  for p, (X, y) in dataloader:
    print(X.size())
    print(y.size())
    print(y)


if __name__ == "__main__":
  test_dataloader()

