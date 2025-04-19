import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class INaturalistDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            inaturalist_full = datasets.ImageFolder(
                root=self.data_dir + "/train", transform=self.transform
            )
            self.inaturalist_train, self.inaturalist_val = random_split(
                inaturalist_full,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )
        elif stage == "test":
            self.inaturalist_test = datasets.ImageFolder(
                root=self.data_dir + "/val", transform=self.transform
            )

        elif stage == "predict":
            self.inaturalist_predict = datasets.ImageFolder(
                root=self.data_dir + "/val", transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.inaturalist_train,
            batch_size=self.batch_size,
            num_workers=11,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.inaturalist_val, batch_size=self.batch_size, num_workers=11
        )

    def test_dataloader(self):
        return DataLoader(
            self.inaturalist_test, batch_size=self.batch_size, num_workers=11
        )

    def predict_dataloader(self):
        return DataLoader(
            self.inaturalist_predict,
            batch_size=self.batch_size,
            shuffle=True,
        )
