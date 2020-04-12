import os
import sys
import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
sys.path.append('../')
import Utils.find_digit as ut

class Ocr(Dataset):
    """
    	dataset class to simplify the data input pipeline
    """
    def __init__(self,usage,transforms):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.usage = usage
        self.transforms = transforms
        self.path = os.path.join("..","Dataset",self.usage)
        self.file_list = os.listdir(self.path)
        self.labels = [int(l.split('_')[0]) for l in self.file_list]
        self.images = []

        for img_path in self.file_list:
            img = cv2.imread(self.path + '/' + img_path,0)
            img = ut.get_digit(img)
            img = cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
            self.images.append(img)
		
    def __len__(self):
    	return len(self.labels) 

    def __getitem__(self,index):
        x = self.transforms(self.images[index])
        y = torch.tensor(self.labels[index],device=self.device,dtype=torch.int64)
        return (x,y)