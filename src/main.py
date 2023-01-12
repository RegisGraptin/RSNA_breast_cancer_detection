
import torch

from src.image_dataset import CustomImageDataset
from src.loss import SmoothBCEwLogits
from src.model import EfficientNetNetwork

import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger


class Config:
    
    RANDOM_SEED = 42
    
    EXPERIMENT_NAME = "efficientnet"
    
    IMAGE_WIDTH  = 224
    IMAGE_HEIGHT = 224
    
    TARGET_CLASS_SIZE = 1
    
    EPOCHS = 10
    BATCH_SIZE = 16
    
    # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    dataset = CustomImageDataset("./train.csv", 
                                        "./train_images_processed_cv2_dicomsdl_256/",
                                        is_dicom=False)

    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=Config.BATCH_SIZE,
                                            shuffle=True, num_workers=6)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE,
                                            shuffle=True, num_workers=6)

    # model = SwinTransformerNetwork(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 1)
    model = EfficientNetNetwork(
        Config.IMAGE_WIDTH, 
        Config.IMAGE_HEIGHT, 
        Config.TARGET_CLASS_SIZE,
        loss = SmoothBCEwLogits(),
    )
    # model = model.to(device)
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    mlf_logger = MLFlowLogger(experiment_name=Config.EXPERIMENT_NAME, tracking_uri="file:./mlruns")
    
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        # limit_train_batches = Config.BATCH_SIZE,
        accelerator = 'gpu',
        devices = 1,
        max_epochs          = Config.EPOCHS,
        logger              = mlf_logger,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    