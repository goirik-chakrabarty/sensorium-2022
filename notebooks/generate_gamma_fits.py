import os
import warnings

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore")

# --- Data Loading and Preparation ---
# This assumes you have downloaded the SENSORIUM data and placed it in the notebooks/data directory
# The filenames list should point to the location of your dataset files.
try:
    from nnfabrik.builder import get_data
except ImportError:
    print(
        "Please make sure you have the nnfabrik library installed: pip install nnfabrik"
    )
    exit()

print("Loading dataset...")
filenames = ["data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip"]
dataset_fn = "sensorium.datasets.static_loaders"
dataset_config = {
    "paths": filenames,
    "normalize": True,
    "include_behavior": False,
    "include_eye_position": False,
    "batch_size": 128,
    "scale": 0.25,
}

# --- This step can be memory-intensive as it loads the full response data ---
print("Extracting full response data...")
path = "data/static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6/data/responses"
n_files = len(os.listdir(path))
responses_list = []
for i in tqdm(range(n_files), desc="Loading response files"):
    res = np.load(os.path.join(path, f"{i}.npy"))
    responses_list.append(np.expand_dims(res, axis=0))

responses = np.concatenate(responses_list, axis=0)
print(f"Response data loaded with shape: {responses.shape}")


# --- Gamma Distribution Fitting ---


class Gamma(nn.Module):
    """
    A PyTorch module to represent a Gamma distribution with learnable parameters.
    """

    def __init__(self):
        super(Gamma, self).__init__()
        # Initialize alpha and beta as learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Use softplus to ensure alpha and beta are always positive
        alpha = torch.nn.functional.softplus(self.alpha)
        beta = torch.nn.functional.softplus(self.beta)
        dist = D.Gamma(concentration=alpha, rate=beta)
        return dist.log_prob(x), alpha.item(), beta.item()


def fit_gamma_distribution(neuron_data, lr=0.05, steps=500, min_threshold=1e-5):
    """
    Fits a Gamma distribution to a single neuron's time series data.

    Args:
        neuron_data (array): 1D array of a neuron's responses over time.
        lr (float): Learning rate for the optimization.
        steps (int): Number of optimization steps.
        min_threshold (float): Values below this threshold are excluded from the fit.

    Returns:
        (alpha, beta): The shape and rate parameters of the fitted Gamma distribution.
    """
    # Filter out NaNs and near-zero values
    x = torch.tensor(neuron_data, dtype=torch.float32)
    x = x[(x > min_threshold) & ~torch.isnan(x)]

    # If there's not enough data, return NaNs
    if len(x) < 10:
        return np.nan, np.nan

    model = Gamma()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optimization loop
    for _ in range(steps):
        optimizer.zero_grad()
        log_probs, _, _ = model(x)
        loss = -log_probs.mean()
        loss.backward()
        optimizer.step()

    # Get the final fitted parameters
    _, alpha, beta = model(x)
    return alpha, beta


# --- Main Execution ---
if __name__ == "__main__":
    print("\nFitting Gamma distributions to each neuron...")
    num_neurons = responses.shape[1]
    gamma_fits = []

    for i in tqdm(range(num_neurons), desc="Fitting neurons"):
        timeseries = responses[:, i]
        alpha, beta = fit_gamma_distribution(timeseries)
        gamma_fits.append((alpha, beta))

    gamma_fits_array = np.array(gamma_fits)

    # --- Save the results ---
    save_path = "/mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/curriculum-learning/sensorium-2022/notebooks/gamma_fits.npy"
    np.save(save_path, gamma_fits_array)
    print(f"\nGamma fit parameters saved to: {save_path}")
    print(f"Shape of the saved array: {gamma_fits_array.shape}")
