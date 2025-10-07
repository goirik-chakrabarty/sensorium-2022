import argparse  # To read command line arguments
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml  # Import the YAML package

warnings.filterwarnings("ignore")

from nnfabrik.builder import get_data, get_model, get_trainer

# --- 1. SET UP ARGUMENT PARSER ---
# This will allow you to pass the config file path from the command line
parser = argparse.ArgumentParser(description="Train a Sensorium model.")
parser.add_argument(
    "--config", type=str, required=True, help="Path to the config.yaml file."
)
args = parser.parse_args()


# --- 2. LOAD CONFIGURATION FROM YAML ---
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Now, get your configurations from the loaded 'config' dictionary
dataset_config = config["dataset_config"]
model_config = config["model_config"]
trainer_config = config["trainer_config"]
model_save_path = config["model_save_path"]


# --- THE REST OF YOUR SCRIPT REMAINS MOSTLY THE SAME ---

# loading the SENSORIUM dataset
# The path is now taken from your config file
filenames = dataset_config["paths"]

# The dataset function remains the same
dataset_fn = "sensorium.datasets.static_loaders"

# Dataloaders are created using the dataset_config from your file
dataloaders = get_data(dataset_fn, dataset_config)

# Instantiate State of the Art Model
model_fn = "sensorium.models.stacked_core_full_gauss_readout"

model = get_model(
    model_fn=model_fn,
    model_config=model_config,  # Using model_config from your file
    dataloaders=dataloaders,
    seed=42,
)

# Configure Trainer
trainer_fn = "sensorium.training.standard_trainer"

# Load gamma_fits from the path specified in the config
trainer_config["gamma_fits"] = np.load(trainer_config["gamma_fits_path"])
del trainer_config["gamma_fits_path"]  # Remove the path from the config


trainer = get_trainer(trainer_fn=trainer_fn, trainer_config=trainer_config)

# Run model training
validation_score, trainer_output, state_dict = trainer(model, dataloaders, seed=42)
print(validation_score)

# Save model checkpoints after training is complete
# The save path is now also from your config file
torch.save(
    model.state_dict(),
    model_save_path,
)
print(f"Model saved to {model_save_path}")
