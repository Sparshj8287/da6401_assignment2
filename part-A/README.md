# Building and Training a CNN from scratch
This directory contains code to build and train a CNN from scratch.

## Usage
Run the training script train.py with the desired command-line arguments. Here's an example command:
```bash
python3 train.py --wandb_entity myname --wandb_project myprojectname --epochs 15 --batch_size 16 --learning_rate 0.002 --conv_layers 5 --filters 32 --filter_stride 1 --filter_size 2 --conv_activation relu --filter_org equal --pooling_alg maxpooling --pooling_size 1 --pooling_stride 3 --dense_size 128 --dense_activation relu --dropout 0.5 --dataset_path  --log_location wandb
```

## Command-Line Arguments

- `--wandb_entity` (`-we`): Wandb entity used to track experiments. Default: `myname`
- `--wandb_project` (`-wp`): Project name in Weights & Biases. Default: `myprojectname`
- `--epochs` (`-e`): Number of epochs to train. Default: `15`
- `--batch_size` (`-b`): Batch size. Default: `16`
- `--learning_rate` (`-lr`): Learning rate. Default: `0.002`
- `--conv_layers` (`-c_l`): Number of convolutional-activation-pooling layers. Default: `5`
- `--filters` (`-f`): Number of filters for the convolutional layers. Default: `32`
- `--filter_stride` (`-f_st`): Filter stride for the convolutional layers. Default: `1`
- `--filter_size` (`-f_s`): Size of the filters for the convolutional layers. Default: `2`
- `--conv_activation` (`-c_a`): Activation for the convolutional layers. Choices: `relu`, `gelu`, `silu`, `mish`. Default: `relu`
- `--filter_org` (`-f_o`): Strategy to modify the number of filters in each convolutional layer. Choices: `equal`, `doubling`, `halving`. Default: `equal`
- `--pooling_alg` (`-p_a`): Pooling algorithm. Choices: `maxpooling`, `avgpooling`. Default: `maxpooling`
- `--pooling_size` (`-p_s`): Kernel size for the pooling layer. Default: `1`
- `--pooling_stride` (`-p_st`): Kernel stride for the pooling layer. Default: `3`
- `--dense_size` (`-d_s`): Number of neurons in the fully connected layer. Default: `128`
- `--dense_activation` (`-d_a`): Activation for the fully connected layers. Choices: `relu`. Default: `relu`
- `--dropout` (`-d`): Dropout value. Default: `0.5`
- `--dataset_path` (`-d_p`): Path to the dataset.
- `--log_location` (`-g`): Log location. Choices: `wandb`, `stdout`. Default: `wandb`

