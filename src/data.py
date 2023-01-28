
import tqdm
from PIL import Image

import glob
import os

from src.analyse.preprocess import MammographyPreprocess

class Config:
    
    IMAGE_WIDTH  = 512
    IMAGE_HEIGHT = 1024
    
    OUTPUT_PATH = "/.../data/train_transformed/"

    DATASET_PATH = "/.../data/train_images/"

if __name__ == "__main__":
    """Script to transform our DICOM files to png files. 

    Allow us to avoid preprocess treatment on our data and to be faster 
    during the training phases.
    """
    
    # Create our preprocess class
    mammographyPreprocess = MammographyPreprocess()
    
    # Create the train_images folder
    train_path = os.path.join(Config.OUTPUT_PATH, "train_images")
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    # For each sub directory
    directories = os.listdir(Config.DATASET_PATH)
    for directory in tqdm.tqdm(directories):
        
        # Get all dicom files
        dcm_path  = os.path.join(Config.DATASET_PATH, directory, "*.dcm")
        dcm_files = glob.glob(dcm_path)
        
        # Create the folder in the new transform dataset
        o_dcm_path = os.path.join(train_path, directory)
        if not os.path.exists(o_dcm_path):
            os.mkdir(o_dcm_path)
        
        # For each file, we do the preprocess
        for file in dcm_files:
            
            filename = "".join(file.split('/')[-1].split('.')[:-1])
            
            extracted_image = mammographyPreprocess.preprocess_mammography(
                file, 
                Config.IMAGE_WIDTH, 
                Config.IMAGE_HEIGHT
            )
            
            outpath  = os.path.join(o_dcm_path, filename + '.png')
            
            im = Image.fromarray(extracted_image)
            im.save(outpath)
            