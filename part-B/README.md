# Finetune a pretrained Model using the INaturalist Dataset
This directory contains code to finetune ResNet50 using the INaturalist Dataset.

## Usage
Run the training script train.py with the desired command-line arguments. Here's an example command:
```bash
python3 train.py --wandb_entity myname --wandb_project myprojectname --model_name mymodel --epochs 15 --batch_size 16 --learning_rate 0.002 --freeze_layers conv1 bn1 --last_layer_strategy replace --dataset_path  --log_location wandb

```

## Command-Line Arguments
- `--wandb_entity` (`-we`): Wandb entity used to track experiments. Default: `myname`
- `--wandb_project` (`-wp`): Project name in Weights & Biases. Default: `myprojectname`
- `--model_name` (`-m_n`): Define the model name.
- `--epochs` (`-e`): Number of epochs to train. Default: `15`
- `--batch_size` (`-b`): Batch size. Default: `16`
- `--learning_rate` (`-lr`): Learning rate. Default: `0.002`
- `--freeze_layers` (`-f_l`): Layers to be frozen. Choices: `conv1`, `bn1`, `relu`, `maxpool`, `layer1`, `layer2`, `layer3`, `layer4`, `avgpool`, `fc`.
- `--last_layer_strategy` (`-l_l_s`): Strategy to change last layer to match output classes. Choices: `replace`, `append`. Default: `replace`
- `--dataset_path` (`-d_p`): Path to the dataset. 
- `--log_location` (`-g`): Log location. Choices: `wandb`, `stdout`. Default: `wandb`

