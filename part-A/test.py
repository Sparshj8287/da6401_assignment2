import torch
from lightning import Trainer
from train import CNN  # assuming your model is saved in model.py
from dataset import INaturalistDataModule
from utils import Activation, FilterOrg, Pooling

# Replace with your actual checkpoint path
checkpoint_path = "/projects/data/astteam/sparsh_assignment2/da6401_assignment2/part-A/checkpoints/cl_5_nfilt_32_bs_16_fo_equal_ca_silu_ds_128_id_ddcrjeg8.ckpt"


# Load the trained weights
model = CNN.load_from_checkpoint(
    checkpoint_path,
    n_layers=5,
    n_filters=32,
    filter_stride=1,
    k=2,
    conv_activation="silu",
    filter_org="equal",
    pooling_alg="maxpooling",
    pooling_size=5,
    pooling_stride=1,
    dense_size=128,
    dense_activation="relu",
    lr=0.0010830652207152835,
    dropout=0.4531594466410468,
)
# Prepare the test dataloader
datamodule = INaturalistDataModule("/projects/data/astteam/sparsh_assignment2/da6401_assignment2/dataset/inaturalist_12K", batch_size=16)
datamodule.setup(stage="test")

# Evaluate on test data
trainer = Trainer()
trainer.test(model=model, datamodule=datamodule)
