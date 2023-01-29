
import torch
import numpy as np
from src.data.image_dataset import CustomImageDataset
from src.model.loss import FocalLoss, SmoothBCEwLogits
from src.model.model import ResNetNetwork


from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler

import pytorch_lightning as pl

from pytorch_lightning.loggers import MLFlowLogger

from sklearn.model_selection import KFold

from torch.utils.data import SubsetRandomSampler, SequentialSampler

class Config:
    
    RANDOM_SEED = 42
    
    EXPERIMENT_NAME = "resnet34"
    
    IMAGE_WIDTH  = 1024
    IMAGE_HEIGHT = 512
        
    TARGET_CLASS_SIZE = 1
    
    EPOCHS = 1
    BATCH_SIZE = 8
    
    K_FOLD = 5
    
    ACCELERATOR = "gpu"
    N_DEVICES = 1
    N_WORKERS = 6

    PATH_DATAFRAME = "/home/rere/data/train_transformed/train.csv"
    PATH_IMAGES    = "/home/rere/data/train_transformed/train_images/"
    DATA_IS_DICOM  = False
    
    LOSS_NAME = "Focal_Loss" 
    MODEL_LOSS = None



class Experiment:

    def __init__(self) -> None:
        pass

    def set_mlflow_logger(self, name: str):
        # Define MLFlow logger
        return MLFlowLogger(
            experiment_name=name, 
            tracking_uri="file:./mlruns"
        )
    
    def get_trainer(self, name_id):
        
        mlflow_logger = self.set_mlflow_logger(f"{Config.EXPERIMENT_NAME}-{name_id}")

        # Trainer configuration
        return pl.Trainer(
            accelerator = Config.ACCELERATOR,
            devices     = Config.N_DEVICES,
            max_epochs  = Config.EPOCHS,
            logger      = mlflow_logger,
        )

    def create_model(self):
        return ResNetNetwork(
            Config.IMAGE_WIDTH, 
            Config.IMAGE_HEIGHT, 
            Config.TARGET_CLASS_SIZE,
            loss = Config.MODEL_LOSS,
        )

    def train_model(self, train_loader, valid_loader, model_path: str = "./", model_id: str = "0"):
        
        model   = self.create_model()
        trainer = self.get_trainer(model_id)
        
        trainer.fit(
            model             = model, 
            train_dataloaders = train_loader, 
            val_dataloaders   = valid_loader
        )
        
        torch.save(model, f"{model_path}{Config.EXPERIMENT_NAME}_{model_id}.ckpt")
        
    def create_dataset(self):
        return CustomImageDataset(Config.PATH_DATAFRAME, 
                                  Config.PATH_IMAGES,
                                  transform = None, # model.transform,
                                  data_augmentation=None, 
                                  width=Config.IMAGE_WIDTH,
                                  height=Config.IMAGE_HEIGHT,
                                  is_dicom=Config.DATA_IS_DICOM)
 
    def kfold_training(self, dataset):
        
        kf = KFold(n_splits = Config.K_FOLD)
        for i, (training_idx, validation_idx) in enumerate(kf.split(dataset)):

            train_sampler = SubsetRandomSampler(training_idx)
            test_sampler  = SequentialSampler(validation_idx)
        
            train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=Config.BATCH_SIZE,
                                                    num_workers=Config.N_WORKERS)
            valid_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=Config.BATCH_SIZE,
                                                    num_workers=Config.N_WORKERS)

            self.train_model(train_loader, valid_loader, model_id=str(i))

    def simple_training(self, dataset):

        Y_values = dataset.y_values()

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        
        np.random.seed(Config.RANDOM_SEED)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_y, val_y = Y_values[train_indices], Y_values[val_indices]

        train_sampler = ExhaustiveWeightedRandomSampler(
            train_y, 
            num_samples = 10000
        )
        # sampler = DistributedProxySampler(
        #     ExhaustiveWeightedRandomSampler(train_y, num_samples=10000)
        # )

        val_sampler  = SequentialSampler(val_indices)

        
        # loader = DataLoader(dataset, sampler=sampler, ...)



        # training_dataset, validation_dataset = torch.utils.data.random_split(
        #     dataset, [0.8, 0.2], generator=torch.Generator()
        # )

        train_loader = torch.utils.data.DataLoader(
            dataset, 
            sampler = train_sampler,
            batch_size=Config.BATCH_SIZE,
            # shuffle=True, 
            num_workers=Config.N_WORKERS
        )
        
        valid_loader = torch.utils.data.DataLoader(
            dataset, 
            sampler = val_sampler,
            batch_size=Config.BATCH_SIZE,
            # shuffle=False, 
            num_workers=Config.N_WORKERS
        )

        print("[*] Length training sets: ", len(train_loader))
        print("[*] Length validation sets: ", len(valid_loader))

        if Config.LOSS_NAME == "Focal_Loss":
            alpha = 0.25
            # n_count = dataset.positive_negative_samples()
            # alpha = n_count[1] / (n_count[1] + n_count[0])
            print("[*] Alpha value: ", str(alpha))
            Config.MODEL_LOSS = FocalLoss(gamma=2.0, alpha=alpha).forward
        
        self.train_model(train_loader, valid_loader, model_id="0")



if __name__ == "__main__":


    # TODO :: WARNING :: Resnet pre trained network resize the data through the model.transform!
    
    experiment      = Experiment()
    model_transform = experiment.create_model().transform

    dataset = experiment.create_dataset()

    print("[*] Size of the dataset: ", len(dataset))
    print(dataset.positive_negative_samples())

    experiment.simple_training(dataset)
        
        