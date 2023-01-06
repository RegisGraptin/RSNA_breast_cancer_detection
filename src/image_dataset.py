
import pandas as pd

import os
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 img_dir: str, 
                 transform=None, 
                 target_transform=None, 
                 is_dicom: bool = False):
        """Custom Image Dataset class.

        Args:
            df (pd.DataFrame): _description_
            img_dir (str): _description_
            transform (_type_, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
            is_dicom (bool, optional): _description_. Defaults to False.
        """
        self.df               = df
        self.img_dir          = img_dir
        self.transform        = transform
        self.target_transform = target_transform
        self.is_dicom         = is_dicom
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir,
                                str(row["patient_id"]), 
                                str(row["image_id"]) + '.dcm')
        
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

training_dataset = CustomImageDataset(df_train, 
                                      "/kaggle/input/rsna-breast-cancer-detection/train_images")

train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=Config.BATCH_SIZE,
                                          shuffle=True, num_workers=2)