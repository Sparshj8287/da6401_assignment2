import argparse
import uuid

import lightning as L
import torch
import torchmetrics
from dataset import INaturalistDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn, optim
from utils import Activation, FilterOrg, Pooling

import wandb


class CNN(L.LightningModule):
    def __init__(
        self,
        n_layers: int,
        n_filters: int,
        filter_stride: int,
        k: int,
        conv_activation: str,
        filter_org: str,
        pooling_alg: str,
        pooling_size: int,
        pooling_stride: int,
        dense_size: int,
        dense_activation: str,
        lr: float,
        dropout: float,
    ):
        """
        Args:
            n_layers (int): Number of convolutional layers.
            n_filters (int): Base number of filters.
            filter_stride (int): Stride for convolution.
            k (int): Kernel size.
            conv_activation (str): Activation function for conv layers.
            filter_org (str): Strategy to organize filters (equal/doubling/halving).
            pooling_alg (str): Pooling method (max/avg).
            pooling_size (int): Pooling kernel size.
            pooling_stride (int): Pooling stride.
            dense_size (int): Size of the fully connected layer.
            dense_activation (str): Activation for the dense layer.
            lr (float): Learning rate.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_filters = n_filters * FilterOrg[filter_org].value
        self.filter_stride = filter_stride
        self.k = k
        self.conv_activation = Activation[conv_activation].value
        self.dense_size = dense_size
        self.filter_org = FilterOrg[filter_org].value
        self.pooling_alg = Pooling[pooling_alg].value
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.dense_activation = Activation[dense_activation].value
        self.lr = lr
        self.dropout = dropout

        self.loss = nn.CrossEntropyLoss()

        # Accuracy metrics for training, validation and test
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

        # Build convolutional and fully connected parts of the network
        self.conv_model: nn.Sequential = self.__build_conv_model()
        inp = torch.zeros(1, 3, 224, 224)
        input_dim = self.conv_model(inp).shape[-1]
        self.fc_model: nn.Sequential = self.__build_fc_model(input_dim)

    def __build_conv_model(self):
        """Builds the convolutional part of the network."""
        model = nn.Sequential()
        num_channels = 3

        for layer in range(self.n_layers):
            model.add_module(
                name=f"conv{layer}",
                module=nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=self.n_filters * self.filter_org,
                    kernel_size=(self.k, self.k),
                    stride=self.filter_stride,
                ),
            )
            model.add_module(f"activation{layer}", self.conv_activation())
            model.add_module(
                f"batch_norm{layer}",
                nn.BatchNorm2d(num_features=self.n_filters * self.filter_org),
            )
            model.add_module(
                f"maxpool{layer}",
                self.pooling_alg(kernel_size=(self.pooling_size, self.pooling_size), stride=self.pooling_stride),
            )
            num_channels = self.n_filters * self.filter_org

        model.add_module("dropout", nn.Dropout2d(p=self.dropout))
        model.add_module("flatten", nn.Flatten())
        return model

    def __build_fc_model(self, input_dim: int):
        """Builds the fully connected part of the network.

        Args:
            input_dim (int): Flattened output dimension from convolutional model.
        """
        model = nn.Sequential()
        model.add_module("linear1", nn.Linear(input_dim, self.dense_size))
        model.add_module("linear_activation1", self.dense_activation())
        model.add_module("linear_batch_norm1", nn.BatchNorm1d(self.dense_size))
        model.add_module("linear2", nn.Linear(self.dense_size, 10))  # 10 classes
        return model

    def forward(self, x):
        """Forward pass."""
        y_conv = self.conv_model(x)
        y = self.fc_model(y_conv)
        return y

    def training_step(self, batch, batch_idx: int):
        """Training step.

        Args:
            batch: A batch of data (inputs and labels).
            batch_idx (int): Index of the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.train_acc(y_hat, y)

        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.val_acc(y_hat, y)

        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.test_acc(y_hat, y)

        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_loss", loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configures optimizer for training."""
        return optim.Adam(self.parameters(), lr=self.lr)


def train(args):
    """Train the model using the given command-line arguments."""
    if args.log_location == "wandb":
        # Initialize W&B logger
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = (
            f"cl_{args.conv_layers}_nfilt_{args.filters}_bs_{args.batch_size}_"
            f"fo_{args.filter_org}_ca_{args.conv_activation}_ds_{args.dense_size}_id_{wandb.run.id}"
        )
        model_name = wandb.run.name
        wandb_logger = WandbLogger(name=args.wandb_entity, log_model=False)
    else:
        model_name = str(uuid.uuid4())
        wandb_logger = None

    print("Training with the following hyperparameters:")
    print(args)

    # Load dataset
    inaturalist = INaturalistDataModule(args.dataset_path, args.batch_size)

    # Initialize model
    model = CNN(
        args.conv_layers,
        args.filters,
        args.filter_stride,
        args.filter_size,
        args.conv_activation,
        args.filter_org,
        args.pooling_alg,
        args.pooling_size,
        args.pooling_stride,
        args.dense_size,
        args.dense_activation,
        args.learning_rate,
        args.dropout,
    )

    # Set up checkpoint saving
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name,
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

    # Train model
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=args.epochs,
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=inaturalist)


if __name__ == "__main__":
    # Parse command-line arguments for training configuration
    parser = argparse.ArgumentParser(description="Train neural network.")
    parser.add_argument("--wandb_entity", "-we", type=str, default="myname", help="Wandb Entity used to track experiments.")
    parser.add_argument("--wandb_project", "-wp", type=str, default="myprojectname", help="Project name in Weights & Biases.")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0010830652207152835, help="Learning rate.")
    parser.add_argument("--conv_layers", "-c_l", type=int, default=5, help="Number of convolutional layers.")
    parser.add_argument("--filters", "-f", type=int, default=32, help="Number of filters for conv layers.")
    parser.add_argument("--filter_stride", "-f_st", type=int, default=1, help="Stride for convolutional layers.")
    parser.add_argument("--filter_size", "-f_s", type=int, default=2, help="Size of the conv filters.")
    parser.add_argument("--conv_activation", "-c_a", type=str, default="silu", choices=["relu", "gelu", "silu", "mish"], help="Activation for conv layers.")
    parser.add_argument("--filter_org", "-f_o", type=str, default="equal", choices=["equal", "doubling", "halving"], help="Filter size change strategy per layer.")
    parser.add_argument("--pooling_alg", "-p_a", type=str, default="maxpooling", choices=["maxpooling", "avgpooling"], help="Pooling algorithm.")
    parser.add_argument("--pooling_size", "-p_s", type=int, default=5, help="Kernel size for pooling.")
    parser.add_argument("--pooling_stride", "-p_st", type=int, default=1, help="Stride for pooling.")
    parser.add_argument("--dense_size", "-d_s", type=int, default=128, help="Size of dense layer.")
    parser.add_argument("--dense_activation", "-d_a", type=str, default="relu", choices=["relu"], help="Activation for dense layer.")
    parser.add_argument("--dropout", "-d", type=float, default=0.4531594466410468, help="Dropout rate.")
    parser.add_argument("--dataset_path", "-d_p", type=str, default="/projects/data/astteam/sparsh_assignment2/da6401_assignment2/dataset/inaturalist_12K", help="Path to dataset.")
    parser.add_argument("--log_location", "-g", type=str, default="wandb", choices=["wandb", "stdout"], help="Log destination.")

    args = parser.parse_args()
    train(args)
