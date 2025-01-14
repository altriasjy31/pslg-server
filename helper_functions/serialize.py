import json
import pickle
import typing as T

def match_by(saving_format: str, obj: T.Any, saving_path: str):
  match saving_format:
    case "pickle":
      with open(saving_path, "wb") as h:
        pickle.dump(obj, h)
    case "json":
      with open(saving_path, "w") as h:
        json.dump(obj, h)