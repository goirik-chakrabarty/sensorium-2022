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
    transfer_trainer_config = config["transfer_trainer_config"]
    model_save_path = config["model_save_path"]

    # --- 2. GET TEST DATALOADERS ---
    dataset_fn = "sensorium.datasets.static_loaders"
    test_dataset_config = dataset_config.copy()

    # --- FIX IS HERE ---
    test_dataset_config["paths"] = test_dataset_config.pop("test_paths")
    if "train_paths" in test_dataset_config:
        del test_dataset_config["train_paths"]  # Remove the unexpected argument

    test_dataloaders = get_data(dataset_fn, test_dataset_config)

    # --- 3. LOAD THE PRE-TRAINED MODEL ---
    model_fn = "sensorium.models.stacked_core_full_gauss_readout"
    model = get_model(
        model_fn=model_fn,
        model_config=model_config,
        dataloaders=test_dataloaders,
        seed=42,
    )

    print(f"Loading pre-trained model from {model_save_path}")
    pretrained_dict = torch.load(model_save_path)
    model.load_state_dict(pretrained_dict, strict=False)

    # --- 4. FREEZE THE CORE ---
    print("Freezing core model parameters...")
    for name, param in model.named_parameters():
        if "readout" not in name:
            param.requires_grad = False

    # --- 5. TRAIN THE NEW READOUT LAYER ---
    print("Training the new readout layer...")
    trainer_fn = "sensorium.training.standard_trainer"
    trainer = get_trainer(trainer_fn=trainer_fn, trainer_config=transfer_trainer_config)

    validation_score, _, _ = trainer(model, test_dataloaders, seed=42, detach_core=True)

    print(f"\nFinished transfer learning.")
    print(f"Final performance on the test set: {validation_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
