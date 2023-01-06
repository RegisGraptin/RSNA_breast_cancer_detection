
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

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
    

class EfficientNetNetwork(nn.Module):

    def __init__(self, width: int, height: int, output_size: int):
        super().__init__()
        self.width       = width
        self.height      = height
        self.output_size = output_size
        
        # Load the model with 1000 outputs
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b2')
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
