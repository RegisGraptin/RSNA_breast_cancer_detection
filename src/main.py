
import torch.optim as optim

import torch

from tqdm import tqdm

from src.image_dataset import CustomImageDataset
from src.loss import SmoothBCEwLogits
from src.model import EfficientNetNetwork

import pytorch_lightning as pl

import mlflow.pytorch
from mlflow import MlflowClient

class Config:
    IMAGE_WIDTH  = 224
    IMAGE_HEIGHT = 224
    
    TARGET_CLASS_SIZE = 1
    
    EPOCHS = 10
    BATCH_SIZE = 16
    
    # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, model_loss, train_generator):
    
    print("Starting training: ", device)
    
    for epoch in range(Config.EPOCHS):
        
        running_loss = 0.0
        for i, (X_batch, Y_batch) in enumerate(tqdm(train_generator), 0):
            # get the inputs; data is a list of [inputs, labels]
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X_batch)
            loss = model_loss(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":

    training_dataset = CustomImageDataset("./train.csv", 
                                        "./train_images_processed_cv2_dicomsdl_256/",
                                        is_dicom=False)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=Config.BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    # model = SwinTransformerNetwork(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, 1)
    model = EfficientNetNetwork(
        Config.IMAGE_WIDTH, 
        Config.IMAGE_HEIGHT, 
        Config.TARGET_CLASS_SIZE,
        loss = SmoothBCEwLogits(),
    )
    model = model.to(device)
    
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        # limit_train_batches = Config.BATCH_SIZE, 
        max_epochs          = Config.EPOCHS
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
    
    