
import numpy as np 
import pandas as pd
import os
import timm

import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from tqdm import tqdm

import torchvision
import torch.nn as nn


from torch.utils.data import Dataset
from torchvision.io import read_image
from skimage.transform import resize

class Config:
    IMAGE_WIDTH  = 224
    IMAGE_HEIGHT = 224
    
    EPOCHS = 10
    BATCH_SIZE = 16
    
    # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].


