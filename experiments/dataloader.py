import typing as T
import torch
import torch as th
import torch.utils.data as ud
import torch.utils.data.dataset as D
import numpy as np

def get_dataloader(dataclass):
  def __wrapper_func(batch_size,
                     dataloader_num_workers,
                     shuffle,
                     validation_split,
                     **datasetconfig):
    dataset = dataclass(**datasetconfig)
    if validation_split > 0:
      random_seed = 42

      # Creating data indices for training and validation splits:
      dataset_size = len(dataset) # type: ignore
      indices = list(range(dataset_size))
      split = int(np.floor(validation_split * dataset_size))
      if shuffle:
          np.random.seed(random_seed)
          np.random.shuffle(indices)
      train_indices, val_indices = indices[split:], indices[:split]

      # Creating PT data samplers and loaders:
      train_sampler = ud.SubsetRandomSampler(train_indices)
      valid_sampler = ud.SubsetRandomSampler(val_indices)

      # shuffle is not compatible with sampler
      train_loader = ud.DataLoader(dataset, batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=dataloader_num_workers)

      val_loader = ud.DataLoader(dataset, batch_size=batch_size,
                                              sampler=valid_sampler,
                                              num_workers=dataloader_num_workers)
    else:
      train_loader = ud.DataLoader(dataset, batch_size=batch_size,
                                                num_workers=dataloader_num_workers,
                                                shuffle=shuffle)
      val_loader = None
    
    if val_loader is None:
      return train_loader
    else:
      assert val_loader is not None
      return train_loader, val_loader
  return __wrapper_func