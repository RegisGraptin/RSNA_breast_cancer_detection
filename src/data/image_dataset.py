
import pandas as pd

import os
import torch
from torch.utils.data import Dataset

from torchvision import transforms

from src.analyse.preprocess import MammographyPreprocess

class CustomImageDataset(Dataset):
    def __init__(self, 
                 df_path: str, 
                 img_dir: str, 
                 transform=None, 
                 data_augmentation=None, 
                 width : int  = 224,
                 height: int = 224,
                 is_dicom: bool = False):
        """Custom Image Dataset class.

        Args:
            df_path (str): Path to the dataframe.
            img_dir (str): Directory of our images.
            transform (_type_, optional): Transformation requiere by our model. Defaults to None.
            data_augmentation (_type_, optional): Transformation used for data augmentation. Defaults to None.
            is_dicom (bool, optional): Indicate if we use RAW images our pre-format images. Defaults to False.
        """
        self.df                = pd.read_csv(df_path)
        self.img_dir           = img_dir
        self.transform         = transform
        self.data_augmentation = data_augmentation
        self.width             = width
        self.height            = height
        self.is_dicom          = is_dicom
        
        self.preprocess = MammographyPreprocess()
        
        # Define the extention of our files
        self.ext = ".dcm"
        if not self.is_dicom:
            self.ext = ".png"
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        # Get the row of our dataframe
        row = self.df.iloc[idx]
        
        # Get the image path
        img_path = os.path.join(self.img_dir,
                                str(row["patient_id"]), 
                                str(row["image_id"]) + self.ext)
        
        # Pre process the image
        image = self.preprocess.preprocess_mammography(img_path, self.width, self.height)
        
        # Get the label
        label = row['cancer']
        
        # Apply data transformation
        if self.data_augmentation:
            image = self.data_augmentation(image)
        
        
        # Apply transformation requiere by our model
        if self.transform:
            
            t = transforms.Compose([
                transforms.ToPILImage(),
            ])

            image = t(image)            
            image = self.transform(image)
        else:
            image = image.reshape((3, self.width, self.height))
            image = torch.from_numpy(image)
            image = image.float()
        
        return image, torch.tensor([label])
