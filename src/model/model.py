
import torch
import torch.nn as nn

import timm

from efficientnet_pytorch import EfficientNet

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pytorch_lightning as pl

class SwinTransformerNetwork(nn.Module):

    HUB_URL    = "SharanSMenon/swin-transformer-hub:main"
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    
    def __init__(self, width: int, height: int, output_size: int):
        super().__init__()
        self.width       = width
        self.height      = height
        self.output_size = output_size
        
        # Load the model with 1000 outputs
        self.pretrained_model = torch.hub.load(SwinTransformerNetwork.HUB_URL, 
                                               SwinTransformerNetwork.MODEL_NAME, 
                                               pretrained=True)
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.out = nn.Sequential(nn.Linear(1000, 100),
                                 nn.BatchNorm1d(100),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(100, self.output_size),
                                 nn.Softmax(dim=0))
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.pretrained_model(image)
        output   = self.out(features)
        return output
    


## Model using EfficientNet
# self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b2')
# self.out = nn.Sequential(nn.Linear(1000, 100),
#                          nn.BatchNorm1d(100),
#                          nn.ReLU(),
#                          nn.Dropout(p=0.2),
#                          nn.Linear(100, self.output_size),
#                          nn.Softmax(dim=0))

    
class ResNetNetwork(pl.LightningModule):

    def __init__(self, width: int, height: int, output_size: int, loss):
        super().__init__()
        
        self.width       = width
        self.height      = height
        self.output_size = output_size
        
        self.loss = loss
        
        self.model = timm.create_model(
            'resnet34',
            pretrained=True, 
            num_classes=self.output_size
        )
        
        # Transform method used from the pretrained network
        self.transform = create_transform(**resolve_data_config(
            self.model.pretrained_cfg, 
            model=self.model)
        )
        
    def transform(self):
        return self.transform
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)
        
    def training_step(self, batch, batch_idx):
        
        X_batch, Y_batch = batch
        
        outputs = self(X_batch)
        
        l = self.loss(outputs, Y_batch)
        
        # Compute accuracy
        acc = (outputs.argmax(dim=-1) == Y_batch).float().mean()

        # Logging to TensorBoard by default
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", l)
        
        return l

    def validation_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        Y_pred = self.model(X_batch)
        l = self.loss(Y_pred, Y_batch)

        labels = Y_pred.argmax(dim=-1)
        acc = (labels == Y_batch).float().mean()

        self.log("val_acc", acc)
        self.log("val_loss", l)

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=0.0001)
