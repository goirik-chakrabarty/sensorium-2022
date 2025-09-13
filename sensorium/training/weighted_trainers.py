import warnings

import numpy as np
import torch
import wandb
from neuralpredictors.training import LongCycler, early_stopping
from tqdm import tqdm

from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss

# This file contains a new trainer function for response magnitude weighting.


def response_magnitude_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    max_iter=200,
    stopper_patience=20,
    lr_decay_steps=3,
    lr_init=0.005,
    max_gradient_norm=1.0,
    use_wandb=False,
    loss_function="PoissonLoss",
    loss_weighting_power=1.0,
    **kwargs,
):
    """
    A trainer that weights the loss function by the magnitude of the true neural response.
    This function is a modified version of the standard_trainer.

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model on
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss by the number of neurons
        max_iter: maximum number of training iterations
        stopper_patience: patience of the early stopping criterion
        lr_decay_steps: how many times to decay the learning rate after early stopping
        lr_init: initial learning rate
        max_gradient_norm: maximum gradient norm for gradient clipping
        use_wandb: whether to use weights and biases for logging
        loss_function: loss function to be used
        loss_weighting_power: The power to which the response magnitudes are raised for weighting.
        **kwargs:
    Returns:
        score: performance of the model on the validation set
        output: user-defined dictionary containing, for example, the model's state dict
        stop_dict: dictionary containing information about the early stopping criterion
    """

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get loss function
    loss_fn = getattr(torch.nn, loss_function)(reduction="none")

    # get optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    # get scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=stopper_patience // 2,
        verbose=True,
        threshold=1e-5,
    )

    # get early stopper, passing the use_wandb flag
    stopper = early_stopping(
        model,
        patience=stopper_patience,
        use_wandb=use_wandb,
        **kwargs,
    )
    score_list = []
    # train over epochs
    for epoch, val_obj in LongCycler(max_iter, stopper, scheduler):

        ########################### Model evaluation ################################################
        model.eval()

        # Compute avg validation and test correlation
        validation_correlation = get_correlations(
            model,
            dataloaders["validation"],
            device=device,
            as_dict=False,
            per_neuron=True,
        )
        np.save(f"./metrics_save/corr/{epoch}.npy", validation_correlation)
        # return the whole tracker output as a dict

        score = np.mean(validation_correlation)
        score_list.append(score)

        wandb.log({"per_epoch_validation_correlation": score})

        ########################### Model training ################################################
        model.train()

        # train over batches
        for data in tqdm(dataloaders["train"], desc="Epoch {}".format(epoch)):

            # zero gradients
            optimizer.zero_grad()

            # move data to device
            data = [d.to(model.device) for d in data]

            # forward pass
            out = model(*data[:-1]) if len(data) > 2 else model(data[0])

            # --- Weighted Loss Calculation ---
            responses = data[1].detach()  # Use true responses for weighting
            # Add a small epsilon to avoid issues with zero responses
            weights = (responses + 1e-8) ** loss_weighting_power

            # Calculate un-reduced loss (per element)
            unweighted_loss = model.loss(out, *data[1:])

            # Apply weights and then take the mean
            loss = (unweighted_loss * weights).mean()

            # backward pass
            loss.backward()

            # gradient clipping
            if max_gradient_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

            # update weights
            optimizer.step()

        # workup on stopper
        if stopper.work_up(val_obj):
            print(f"Early stopping at epoch {epoch}.")

    # return best score and output
    return stopper.best_score, stopper.output, stopper.stop_dict
