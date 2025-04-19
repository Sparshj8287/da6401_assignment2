import argparse
import uuid
from typing import List

import lightning as L
import torchmetrics
from dataset import INaturalistDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn, optim
from torchvision.models import ResNet50_Weights, resnet50

import wandb


class ResNet50(L.LightningModule):
    def __init__(
        self,
        freeze_layers: List[str],
        last_layer_strategy: str,
        lr: float,
    ):
        """
        Args:
            freeze_layers (List[str]): Layers of the backbone to freeze.
            last_layer_strategy (str): Strategy to modify the last layer ('replace' or 'append').
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.freeze_layers = freeze_layers
        self.last_layer_strategy = last_layer_strategy
        self.lr = lr

        self.loss = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

        # Load pretrained ResNet50
        self.resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.new_model = self.__build_model()

    def __build_model(self):
        """Modifies the ResNet50 model based on freeze and last layer strategy."""
        new_model = nn.Sequential()

        # Freeze specified layers
        for k in self.resnet_model._modules.keys():
            if k in self.freeze_layers:
                self.resnet_model._modules[k].requires_grad_(False)

        # Modify the last layer
        if self.last_layer_strategy == "replace":
            self.resnet_model._modules["fc"] = nn.Identity()
            new_model.add_module(
                name="fc", module=nn.Linear(in_features=2048, out_features=10)
            )
        else:
            new_model.add_module(name="fc_act", module=nn.ReLU())
            new_model.add_module(
                name="fc2", module=nn.Linear(in_features=1000, out_features=10)
            )

        return new_model

    def forward(self, x):
        """Forward pass through base model and final classification head."""
        y1 = self.resnet_model(x)
        y = self.new_model(y1)
        return y

    def training_step(self, batch, batch_idx: int):
        """
        Performs a training step.

        Args:
            batch: A batch of training data.
            batch_idx (int): Index of the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        # Track accuracy
        self.train_acc(y_hat, y)

        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step.

        Args:
            batch: A batch of validation data.
            batch_idx (int): Index of the batch.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        # Track validation accuracy
        self.val_acc(y_hat, y)

        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Sets up optimizer (Adam)."""
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train(args):
    """
    Sets up data, model, logger, and trainer, and starts training.

    Args:
        args: Parsed command line arguments.
    """
    # Initialize WandB logger if specified
    if args.log_location == "wandb":
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = args.model_name
        model_name = args.model_name
        wandb_logger = WandbLogger(name=args.wandb_entity, log_model=False)
    else:
        model_name = str(uuid.uuid4())
        wandb_logger = None

    print("Training with the following hyperparameters:")
    print(args)

    # Load dataset
    inaturalist = INaturalistDataModule(args.dataset_path, args.batch_size)

    # Initialize model
    model = ResNet50(
        args.freeze_layers,
        args.last_layer_strategy,
        args.learning_rate,
    )

    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name,
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

    # Configure trainer
    trainer = L.Trainer(
        limit_train_batches=100,
        max_epochs=args.epochs,
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    # Start training
    trainer.fit(model=model, datamodule=inaturalist)


if __name__ == "__main__":
    # Define CLI arguments for training
    parser = argparse.ArgumentParser(description="Finetune ResNet50 network.")

    parser.add_argument(
        "--wandb_entity", "-we", type=str, default="sjshiva8287",
        help="Wandb Entity used to track experiments.",
    )
    parser.add_argument(
        "--wandb_project", "-wp", type=str, default="da6401_assignment2",
        help="Project name in Weights & Biases.",
    )
    parser.add_argument(
        "--model_name", "-m_n", type=str,
        help="Define the model name",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=15,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.002,
        help="Learning rate.",
    )
    parser.add_argument(
        "--freeze_layers", "-f_l", nargs="+",
        choices=[
            "conv1", "bn1", "relu", "maxpool",
            "layer1", "layer2", "layer3", "layer4",
            "avgpool", "fc",
        ],
        default=[
            "conv1", "bn1", "relu", "maxpool",
            "layer1", "layer2", "layer3", "layer4",
            "avgpool", "fc",
        ],
        help="Layers to be frozen",
    )
    parser.add_argument(
        "--last_layer_strategy", "-l_l_s", type=str, default="replace",
        choices=["replace", "append"],
        help="Strategy to change last layer to match output classes",
    )
    parser.add_argument(
        "--dataset_path", "-d_p", type=str,
        default="/projects/data/astteam/sparsh_assignment2/da6401_assignment2/dataset/inaturalist_12K",
        help="Path to dataset",
    )
    parser.add_argument(
        "--log_location", "-g", type=str, default="wandb",
        choices=["wandb", "stdout"],
        help="Log location",
    )

    args = parser.parse_args()
    train(args)
