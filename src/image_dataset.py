
import pandas as pd

import os
import torch
from torch.utils.data import Dataset

from src.utils import open_dicom_image, open_png_image

class CustomImageDataset(Dataset):
    def __init__(self, 
                 df_path: str, 
                 img_dir: str, 
                 transform=None, 
                 target_transform=None, 
                 is_dicom: bool = False):
        """Custom Image Dataset class.

        Args:
            df_path (str): _description_
            img_dir (str): _description_
            transform (_type_, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
            is_dicom (bool, optional): _description_. Defaults to False.
        """
        self.df               = pd.read_csv(df_path)
        self.img_dir          = img_dir
        self.transform        = transform
        self.target_transform = target_transform
        self.is_dicom         = is_dicom
        self.ext = ".dcm"
        
        if not self.is_dicom:
            self.ext = ".png"
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir,
                                str(row["patient_id"]), 
                                str(row["image_id"]) + self.ext)
        
        if self.is_dicom:    
           image = open_dicom_image(img_path)
        else:
            image = open_png_image(img_path)
            
        label = row['cancer']
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            image = self.target_transform(image)
        
        return torch.from_numpy(image), torch.tensor([label])
