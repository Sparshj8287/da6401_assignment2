import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class INaturalistDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        """
        Args:
            data_dir (str): Directory where the dataset is located.
            batch_size (int): Number of samples per batch to load.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define standard transformations: resizing, normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def setup(self, stage: str):
        """
        Prepares datasets for different stages (fit, test, predict).
        
        Args:
            stage (str): Current stage - 'fit', 'test', or 'predict'.
        """
        if stage == "fit":
            # Load full training dataset and split into train and validation
            inaturalist_full = datasets.ImageFolder(
                root=self.data_dir + "/train", transform=self.transform
            )
            self.inaturalist_train, self.inaturalist_val = random_split(
                inaturalist_full,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
        elif stage == "test":
            # Load validation set for testing
            self.inaturalist_test = datasets.ImageFolder(
                root=self.data_dir + "/val", transform=self.transform
            )
        elif stage == "predict":
            # Load validation set for predictions
            self.inaturalist_predict = datasets.ImageFolder(
                root=self.data_dir + "/val", transform=self.transform
            )

    def train_dataloader(self):
        """
        Returns training data loader.
        """
        return DataLoader(
            self.inaturalist_train,
            batch_size=self.batch_size,
            num_workers=11,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Returns validation data loader.
        """
        return DataLoader(
            self.inaturalist_val, batch_size=self.batch_size, num_workers=11
        )

    def test_dataloader(self):
        """
        Returns test data loader.
        """
        return DataLoader(
            self.inaturalist_test, batch_size=self.batch_size, num_workers=11
        )

    def predict_dataloader(self):
        """
        Returns prediction data loader.
        """
        return DataLoader(
            self.inaturalist_predict,
            batch_size=self.batch_size,
            shuffle=True,
        )
