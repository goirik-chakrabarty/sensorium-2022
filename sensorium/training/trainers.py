import math
import os
import warnings
from functools import partial

import numpy as np
import torch
import wandb
from neuralpredictors.measures import modules
from neuralpredictors.training import (
    LongCycler,
    MultipleObjectiveTracker,
    early_stopping,
)
from nnfabrik.utility.nn_helpers import set_random_seed
from scipy.special import lambertw
from torch import nn
from tqdm import tqdm

from ..utility import scores
from ..utility.scores import get_correlations, get_poisson_loss

ITER = 0


class PoissonLoss(nn.Module):
    def __init__(self, bias=1e-08, per_neuron=False, avg=True, full_loss=False):
        """
        Computes Poisson loss between the output and target. Loss is evaluated by computing log likelihood that
        output prescribes the mean of the Poisson distribution and target is a sample from the distribution.

        Args:
            bias (float, optional): Value used to numerically stabilize evalution of the log-likelihood. This value is effecitvely added to the output during evaluation. Defaults to 1e-08.
            per_neuron (bool, optional): If set to True, the average/total Poisson loss is returned for each entry of the last dimension (assumed to be enumeration neurons) separately. Defaults to False.
            avg (bool, optional): If set to True, return mean loss. Otherwise returns the sum of loss. Defaults to True.
            full_loss (bool, optional): If set to True, compute the full loss, i.e. with Stirling correction term (not needed for optimization but needed for reporting of performance). Defaults to False.
        """
        super().__init__()
        self.bias = bias
        self.full_loss = full_loss
        self.per_neuron = per_neuron
        self.avg = avg
        if self.avg:
            warnings.warn(
                "Poissonloss is averaged per batch. It's recommended to use `sum` instead"
            )

    def forward(self, output, target):
        global ITER
        ITER += 1
        target = target.detach()
        rate = output
        loss = nn.PoissonNLLLoss(
            log_input=False, full=self.full_loss, eps=self.bias, reduction="none"
        )(rate, target)
        wandb.log({"poisson_loss": loss.sum().item()})

        save_dir = "./metrics_save/loss/"
        os.makedirs(save_dir, exist_ok=True)

        np.save(f"./metrics_save/loss/{ITER}", loss.sum(dim=0).cpu().detach().numpy())

        # if True:# self.elementwise:
        #     return loss.sum(dim=0) # dim = 1 or 0 # SWEEP
        if not self.per_neuron:
            loss = loss.mean() if self.avg else loss.sum()
        else:
            loss = loss.view(-1, loss.shape[-1])
            loss = loss.mean(dim=0) if self.avg else loss.sum(dim=0)
        assert not (
            torch.isnan(loss).any() or torch.isinf(loss).any()
        ), "None or inf value encountered!"
        return loss


class SuperLoss(nn.Module):

    def __init__(self, loss_fn=None, C=10, lam=0.1):
        super(SuperLoss, self).__init__()
        self.lam = lam
        self.counter = 0
        self.tau = 0

    def forward(self, loss):
        self.counter += 1
        l_i = loss.detach()
        self.tau = (self.tau * (self.counter - 1) + l_i.mean()) / self.counter
        sigma = self.sigma(l_i)
        loss = (loss - self.tau) * sigma + self.lam * (torch.log(sigma) ** 2)
        return loss

    def sigma(self, l_i):
        x = torch.ones(l_i.size()) * (-2 / math.exp(1.0))
        x = x.cuda()
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).cuda()
        return sigma


def standard_trainer(
    model,
    dataloaders,
    seed,
    use_wandb=True,
    avg_loss=False,
    scale_loss=False,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    track_training=False,
    detach_core=False,
    loss_weighting_power=None,  # --- ADDED: New argument for reweighting
    **kwargs,
):
    """

    Args:
        model: model to be trained
        ...
        loss_weighting_power: The power to which the response magnitudes are raised for weighting.
        ...
    """

    def full_objective(model, dataloader, data_key, *args, **kwargs):
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )
        # --- MODIFIED: Loss calculation to include reweighting ---
        unweighted_loss = criterion(
            model(args[0].to(device), data_key=data_key, **kwargs),
            args[1].to(device),
        )

        if loss_weighting_power is not None and loss_weighting_power > 0:
            responses = args[1].to(device).detach()
            # Calculate raw weights
            weights = (responses + 1e-8) ** loss_weighting_power

            # This keeps the total loss magnitude similar to the unweighted case on average.
            weights = weights / (weights.mean() + 1e-8)

            loss = unweighted_loss * weights
        else:
            loss = unweighted_loss

        regularizers = int(
            not detach_core
        ) * model.core.regularizer() + model.readout.regularizer(data_key)

        final_loss = (loss_scale * loss).sum() + regularizers
        wandb.log({"loss": final_loss.item()})
        return final_loss

    wandb.init(
        project="curriculum-learning",
        config={
            "seed": seed,
            "avg_loss": avg_loss,
            "scale_loss": scale_loss,
            "loss_function": loss_function,
            "stop_function": stop_function,
            "loss_accum_batch_n": loss_accum_batch_n,
            "device": device,
            "verbose": verbose,
            "interval": interval,
            "patience": patience,
            "epoch": epoch,
            "lr_init": lr_init,
            "max_iter": max_iter,
            "maximize": maximize,
            "tolerance": tolerance,
            "restore_best": restore_best,
            "lr_decay_steps": lr_decay_steps,
            "lr_decay_factor": lr_decay_factor,
            "min_lr": min_lr,
            "loss_weighting_power": loss_weighting_power,
        },
        mode="online" if use_wandb else "disabled",
    )

    ##### Model training #########################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

    # criterion = getattr(modules, loss_function)(avg=avg_loss)
    criterion = PoissonLoss(avg=avg_loss)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["validation"],
        device=device,
        per_neuron=False,
        avg=True,
    )

    n_iterations = len(LongCycler(dataloaders["train"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = (
        len(dataloaders["train"].keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )

    if track_training:
        tracker_dict = dict(
            correlation=partial(
                get_correlations,
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
            ),
            poisson_loss=partial(
                get_poisson_loss,
                model,
                dataloaders["validation"],
                device=device,
                per_neuron=False,
                avg=False,
            ),
        )
        if hasattr(model, "tracked_values"):
            tracker_dict.update(model.tracked_values)
        tracker = MultipleObjectiveTracker(**tracker_dict)
    else:
        tracker = None

    # train over epochs
    # for epoch, val_obj in early_stopping(
    #     model,
    #     stop_closure,
    #     interval=interval,
    #     patience=patience,
    #     start=epoch,
    #     max_iter=max_iter,
    #     maximize=maximize,
    #     tolerance=tolerance,
    #     restore_best=restore_best,
    #     tracker=tracker,
    #     scheduler=scheduler,
    #     lr_decay_steps=lr_decay_steps,
    # ):
    score_list = []
    for epoch in range(max_iter):
        ########################### Model evaluation ################################################
        model.eval()
        tracker.finalize() if track_training else None

        # Compute avg validation and test correlation
        validation_correlation = get_correlations(
            model,
            dataloaders["validation"],
            device=device,
            as_dict=False,
            per_neuron=True,
        )
        save_dir = "./metrics_save/corr"
        save_path = os.path.join(save_dir, f"{epoch}.npy")

        os.makedirs(save_dir, exist_ok=True)

        if os.path.exists(save_path):
            print(f"Warning: {save_path} already exists and will be overwritten.")

        # Save file
        np.save(save_path, validation_correlation)
        print(f"Saved validation_correlation to {save_path}")

        # return the whole tracker output as a dict
        output = {k: v for k, v in tracker.log.items()} if track_training else {}
        output["validation_corr"] = validation_correlation

        score = np.mean(validation_correlation)
        score_list.append(score)

        wandb.log({"per_epoch_validation_correlation": score})

        ########################### Model training ################################################
        model.train()
        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):

            batch_args = list(data)
            batch_kwargs = data._asdict() if not isinstance(data, dict) else data
            loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )
            loss.backward()
            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad()

    wandb.log({"final_validation_correlation": score})
    wandb.log({"max_validation_correlation": max(score_list)})

    return score, output, model.state_dict()
