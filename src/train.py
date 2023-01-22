
import torch

from src.data.image_dataset import CustomImageDataset
from src.model.loss import SmoothBCEwLogits
from src.model.model import ResNetNetwork

import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger

from sklearn.model_selection import KFold

from torch.utils.data import SubsetRandomSampler, SequentialSampler

class Config:
    
    RANDOM_SEED = 42
    
    EXPERIMENT_NAME = "resnet34"
    
    IMAGE_WIDTH  = 224
    IMAGE_HEIGHT = 224
        
    TARGET_CLASS_SIZE = 1
    
    EPOCHS = 1
    BATCH_SIZE = 16
    
    K_FOLD = 5
    
    ACCELERATOR = "gpu"
    N_DEVICES = 1
    N_WORKERS = 6
    
    


if __name__ == "__main__":

    # model = SwinTransformerNetwork(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 1)
    model = ResNetNetwork(
        Config.IMAGE_WIDTH, 
        Config.IMAGE_HEIGHT, 
        Config.TARGET_CLASS_SIZE,
        loss = SmoothBCEwLogits(),
    )
    
    # Define MLFlow logger
    
    mlf_logger = MLFlowLogger(
        experiment_name=Config.EXPERIMENT_NAME, 
        tracking_uri="file:./mlruns"
    )
        
    dataset = CustomImageDataset("./data/train.csv", 
                                 "./data/train_images_processed_cv2_dicomsdl_256/",
                                 transform = model.transform,
                                 is_dicom  = False)

    kf = KFold(n_splits = Config.K_FOLD)
    for i, (training_idx, validation_idx) in enumerate(kf.split(dataset)):

        train_sampler = SubsetRandomSampler(training_idx)
        test_sampler  = SequentialSampler(validation_idx)
    

        # training_dataset, validation_dataset = torch.utils.data.random_split(
        #     dataset, [0.8, 0.2], generator=torch.Generator()
        # )

        train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=Config.BATCH_SIZE,
                                                num_workers=Config.N_WORKERS)
        valid_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=Config.BATCH_SIZE,
                                                num_workers=Config.N_WORKERS)

        
        # Trainer configuration
        trainer = pl.Trainer(
            # limit_train_batches = Config.BATCH_SIZE,
            accelerator = Config.ACCELERATOR,
            devices     = Config.N_DEVICES,
            max_epochs  = Config.EPOCHS,
            logger      = mlf_logger,
        )
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=valid_loader
        )
        
        torch.save(model, f"./model_{i}.ckpt")
        
        
        