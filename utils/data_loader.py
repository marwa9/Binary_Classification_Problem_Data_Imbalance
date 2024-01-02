#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd 
from PIL import Image as PImage
import torchvision.transforms as transforms
from torch.utils import data


class Dataset(data.Dataset):
  def __init__(self, split_file,data_path,training):
    self.list_IDs = []
    self.labels = []
    df = pd.read_csv(split_file,index_col=0)
    for el in list(df['sample']):
        self.list_IDs.append(os.path.join(data_path,el))
    self.labels = df['label']
    self.training = training

  def __len__(self):
      return len(self.list_IDs)

  def __getitem__(self, index):
      
      if self.training:
          data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                                transforms.ToTensor()])
      else:
          data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.ToTensor()])
      return data_transforms(PImage.open(self.list_IDs[index])),self.labels[index]
