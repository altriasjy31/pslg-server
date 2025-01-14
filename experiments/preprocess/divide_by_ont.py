import pickle
import json

import pandas as pd
import typing as T
import operator as opr

import os
import sys
prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if prj_dir not in sys.path:
  sys.path.append(prj_dir)
import helper_functions.obo_parser as op
import helper_functions.serialize as serialize


OntDict = T.Dict[T.Union[str, T.Hashable], T.List[str]]
ResultDict = T.Dict[str, OntDict]

def pandas_to_dict(data: pd.DataFrame):
  """
  Returns
   Dict{"cc": Dict{"proteins": List[str], 
                   "prop_annotations": List[str]}
        "mf": ...
        "bp:: ...}
  """
  assert "proteins" in data.columns, \
    "a column named proteins not in this dataframe"
  
  assert "prop_annotations" in data.columns, \
    "a column named prop_annotations not in this dataframe"

  ont_masks: pd.Series         
  ont_masks = data["prop_annotations"].apply(lambda p: (op.CELLULAR_COMPONENT in p,
                                                        op.MOLECULAR_FUNCTION in p,
                                                        op.BIOLOGICAL_PROCESS in p))
  
  gen_by_mask: T.Callable[[int], OntDict]
  gen_by_mask = lambda x: (p := data[["proteins", "prop_annotations"]],
                           p1 := p[ont_masks.apply(opr.itemgetter(x))],
                           p1.to_dict("list"))[-1]
  return {
    "cellular_component": gen_by_mask(0),
    "molecular_function": gen_by_mask(1),
    "biological_process": gen_by_mask(2)
  }

if __name__ == "__main__":
  import argparse as argp  

  parser = argp.ArgumentParser()

  parser.add_argument("data", type=str, help="a pickle format file that saved by pandas dataframe")
  parser.add_argument("saving_path", type=str)

  parser.add_argument("-f", "--saving-format",
                      dest="format",
                      default="pickle")

  opt = parser.parse_args()

  with open(opt.data, "rb") as h:
    data = pickle.load(h)
  
  result_dict = pandas_to_dict(data)

  saving_format: str = opt.format
  serialize.match_by(saving_format, result_dict, opt.saving_path)