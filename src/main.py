
import torch

from src.image_dataset import CustomImageDataset
from src.loss import SmoothBCEwLogits
from src.model import ResNetNetwork

import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger


class Config:
    
    RANDOM_SEED = 42
    
    EXPERIMENT_NAME = "efficientnet"
    
    IMAGE_WIDTH  = 224
    IMAGE_HEIGHT = 224
    
    TARGET_CLASS_SIZE = 1
    
    EPOCHS = 10
    BATCH_SIZE = 16
    
    ACCELERATOR = "gpu"
    N_DEVICES = 1


if __name__ == "__main__":

    # model = SwinTransformerNetwork(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 1)
    model = ResNetNetwork(
        Config.IMAGE_WIDTH, 
        Config.IMAGE_HEIGHT, 
        Config.TARGET_CLASS_SIZE,
        loss = SmoothBCEwLogits(),
    )
        
    dataset = CustomImageDataset("./data/train.csv", 
                                 "./data/train_images_processed_cv2_dicomsdl_256/",
                                 transform = model.transform,
                                 is_dicom  = False)

    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=Config.BATCH_SIZE,
                                            shuffle=True, num_workers=6)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE,
                                            shuffle=True, num_workers=6)

    
    # Define MLFlow logger
    mlf_logger = MLFlowLogger(
        experiment_name=Config.EXPERIMENT_NAME, 
        tracking_uri="file:./mlruns"
    )
    
    # Trainer configuration
    trainer = pl.Trainer(
        # limit_train_batches = Config.BATCH_SIZE,
        accelerator = Config.ACCELERATOR,
        devices     = Config.N_DEVICES,
        max_epochs  = Config.EPOCHS,
        logger      = mlf_logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    