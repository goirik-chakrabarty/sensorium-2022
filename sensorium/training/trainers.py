import math
import os
import random
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
from scipy.stats import gamma as gamma_dist
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


def rand_bbox(size, lam):
    """Generates a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def standard_trainer(
    model,
    dataloaders,
    seed,
    use_wandb=True,
    use_tqdm=True,
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
    loss_weighting_power=None,
    use_performance_tail_weighting=False,
    gamma_fits=None,
    tail_quantile=0.95,
    use_mixup=False,
    mixup_alpha=0.4,
    use_cutmix=False,
    cutmix_alpha=1.0,
    use_manifold_mixup=False,
    manifold_mixup_alpha=0.4,
    manifold_mixup_layer_pool=None,
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders
        seed: random seed
        use_wandb: whether to use weights and biases
        use_tqdm: whether to use tqdm
        avg_loss: whether to average the loss
        scale_loss: whether to scale the loss
        loss_function: loss function to use
        stop_function: function to use for early stopping
        loss_accum_batch_n: number of batches to accumulate loss over
        device: device to use
        verbose: whether to print progress
        interval: interval for early stopping
        patience: patience for early stopping
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of iterations
        maximize: whether to maximize the score
        tolerance: tolerance for early stopping
        restore_best: whether to restore the best model
        lr_decay_steps: number of learning rate decay steps
        lr_decay_factor: learning rate decay factor
        min_lr: minimum learning rate
        cb: callback function
        track_training: whether to track training
        detach_core: whether to detach the core
        loss_weighting_power: power to use for loss weighting
        use_performance_tail_weighting: whether to use performance tail weighting
        gamma_fits: gamma fits for performance tail weighting
        tail_quantile: quantile to use for performance tail weighting
        use_mixup: whether to use mixup
        mixup_alpha: alpha for mixup
        use_cutmix: whether to use cutmix
        cutmix_alpha: alpha for cutmix
        use_manifold_mixup: If True, enables Manifold Mixup data augmentation.
        manifold_mix_alpha: The alpha parameter for the Beta distribution used in Manifold Mixup.
        manifold_mix_layer_pool: A list of layer indices from which to randomly select for applying Manifold Mixup.
        **kwargs:
    """

    if use_performance_tail_weighting:
        if gamma_fits is None:
            raise ValueError(
                "gamma_fits must be provided when use_performance_tail_weighting is True."
            )
        n_neurons = gamma_fits.shape[0]
        neuron_performance = torch.ones(n_neurons, device=device, dtype=torch.float32)
    else:
        neuron_performance = None

    def full_objective(model, dataloader, data_key, *args, **kwargs):

        nonlocal neuron_performance
        loss_scale = (
            np.sqrt(len(dataloader[data_key].dataset) / args[0].shape[0])
            if scale_loss
            else 1.0
        )

        inputs, targets = args[0].to(device), args[1].to(device)

        r = np.random.rand(1)
        if use_mixup and use_cutmix:
            if r < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, mixup_alpha, device
                )
                outputs = model(inputs, data_key=data_key, **kwargs)
                unweighted_loss = mixup_criterion(
                    criterion, outputs, targets_a, targets_b, lam
                )
            else:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                rand_index = torch.randperm(inputs.size()[0]).to(device)
                targets_a = targets
                targets_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[
                    rand_index, :, bbx1:bbx2, bby1:bby2
                ]
                lam = 1 - (
                    (bbx2 - bbx1)
                    * (bby2 - bby1)
                    / (inputs.size()[-1] * inputs.size()[-2])
                )

                outputs = model(inputs, data_key=data_key, **kwargs)
                unweighted_loss = mixup_criterion(
                    criterion, outputs, targets_a, targets_b, lam
                )
        elif use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, mixup_alpha, device
            )
            outputs = model(inputs, data_key=data_key, **kwargs)
            unweighted_loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam
            )
        elif use_cutmix:
            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            targets_a = targets
            targets_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2])
            )

            outputs = model(inputs, data_key=data_key, **kwargs)
            unweighted_loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam
            )
        elif use_manifold_mixup:
            lam = np.random.beta(manifold_mixup_alpha, manifold_mixup_alpha)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            targets_a = targets
            targets_b = targets[rand_index]

            layer_to_mix = random.choice(manifold_mixup_layer_pool)

            pre_mix_model = nn.Sequential(
                *list(model.core.features.children())[:layer_to_mix]
            )
            post_mix_model = nn.Sequential(
                *list(model.core.features.children())[layer_to_mix:]
            )

            h1 = pre_mix_model(inputs)
            h2 = pre_mix_model(inputs[rand_index])

            mixed_h = lam * h1 + (1 - lam) * h2

            core_output = post_mix_model(mixed_h)
            outputs = model.readout(core_output, data_key=data_key)

            unweighted_loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam
            )
        else:
            outputs = model(inputs, data_key=data_key, **kwargs)
            unweighted_loss = criterion(outputs, targets)

        responses = targets.detach()

        if use_performance_tail_weighting:
            if gamma_fits is None:
                raise ValueError(
                    "gamma_fits must be provided when use_performance_tail_weighting is True."
                )

            with torch.no_grad():
                alphas = torch.tensor(
                    gamma_fits[:, 0], device=device, dtype=torch.float32
                )
                betas = torch.tensor(
                    gamma_fits[:, 1], device=device, dtype=torch.float32
                )

                tail_thresholds = torch.tensor(
                    gamma_dist.ppf(
                        tail_quantile,
                        a=alphas.cpu().numpy(),
                        scale=1 / betas.cpu().numpy(),
                    ),
                    device=device,
                    dtype=torch.float32,
                )

                is_in_tail = responses > tail_thresholds
                weight_if_high_performance = 1.0 / (neuron_performance + 1e-8)
                weight_if_low_performance = torch.full_like(neuron_performance, 1000.0)

                performance_weights = torch.where(
                    neuron_performance >= 0.001,
                    weight_if_high_performance,
                    weight_if_low_performance,
                )

                # --- Create and normalize weights ---
                weights = torch.ones_like(responses)
                weights[is_in_tail] = performance_weights.expand_as(responses)[
                    is_in_tail
                ]
                weights = weights / (weights.mean() + 1e-8)

            loss = unweighted_loss * weights

            global ITER
            if ITER % 35 == 1 and not use_tqdm:
                arr = neuron_performance.cpu().numpy()
                ind = np.argpartition(arr, -5)[-5:]
                ind = ind[np.argsort(arr[ind])]
                rev_ind = np.argpartition(arr, 5)[:5]
                rev_ind = rev_ind[np.argsort(arr[rev_ind])]
                print(rev_ind, "|", ind)
                print(np.around(arr[rev_ind], 2), "|", np.around(arr[ind], 2))
                print(
                    np.around(performance_weights[rev_ind].cpu().numpy(), 2),
                    "|",
                    np.around(performance_weights[ind].cpu().numpy(), 2),
                )
                print(f"Average correlation epoch {ITER//35} : {arr.mean()}")
                print(f"Average correlation epoch {ITER//35} : {arr[6477]}")
                print(f"Average correlation epoch {ITER//35} : {arr[6151]}")

        elif loss_weighting_power is not None and loss_weighting_power > 0:
            weights = (responses + 1e-8) ** loss_weighting_power
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
            "use_performance_tail_weighting": use_performance_tail_weighting,
            "tail_quantile": tail_quantile,
            "use_mixup": use_mixup,
            "mixup_alpha": mixup_alpha,
            "use_cutmix": use_cutmix,
            "cutmix_alpha": cutmix_alpha,
            "use_manifold_mixup": use_manifold_mixup,
            "manifold_mixup_alpha": manifold_mixup_alpha,
            "manifold_mixup_layer_pool": manifold_mixup_layer_pool,
        },
        mode="online" if use_wandb else "disabled",
    )

    ##### Model training #########################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()

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

        validation_correlation = get_correlations(
            model,
            dataloaders["validation"],
            device=device,
            as_dict=False,
            per_neuron=True,
        )

        if use_performance_tail_weighting:
            neuron_performance = torch.tensor(
                validation_correlation, device=device, dtype=torch.float32
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
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        if cb is not None:
            cb()

        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(
            enumerate(LongCycler(dataloaders["train"])),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
            disable=not (use_tqdm),
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
