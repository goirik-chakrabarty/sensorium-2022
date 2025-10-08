import argparse
import os
import warnings

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

from nnfabrik.builder import get_data, get_model, get_trainer


def main(config_path):
    # --- 1. LOAD CONFIGURATION ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    trainer_config = config["trainer_config"]
    model_save_path = config["model_save_path"]

    # --- 2. GET TRAINING DATALOADERS ---
    dataset_fn = "sensorium.datasets.static_loaders"
    train_dataset_config = dataset_config.copy()

    # --- FIX IS HERE ---
    train_dataset_config["paths"] = train_dataset_config.pop("train_paths")
    if "test_paths" in train_dataset_config:
        del train_dataset_config["test_paths"]  # Remove the unexpected argument

    train_dataloaders = get_data(dataset_fn, train_dataset_config)

    # --- 3. BUILD AND TRAIN THE MODEL ---
    model_fn = "sensorium.models.stacked_core_full_gauss_readout"
    model = get_model(
        model_fn=model_fn,
        model_config=model_config,
        dataloaders=train_dataloaders,
        seed=42,
    )

    trainer_fn = "sensorium.training.standard_trainer"
    if "gamma_fits_path" in trainer_config:
        trainer_config["gamma_fits"] = np.load(trainer_config["gamma_fits_path"])
        del trainer_config["gamma_fits_path"]

    trainer = get_trainer(trainer_fn=trainer_fn, trainer_config=trainer_config)

    # Train the model
    validation_score, _, _ = trainer(model, train_dataloaders, seed=42)
    print(f"Finished training. Final validation score: {validation_score}")

    # --- 4. SAVE THE TRAINED MODEL ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
