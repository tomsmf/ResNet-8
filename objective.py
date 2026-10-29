"""The black-box objective f(config, seed) -> score: train ResNet-8, return test accuracy."""

import math
import time

import torch
import torch.nn as nn

import data
from model import ResNet8
from search_space import SGD_WEIGHT_DECAY


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_optimizer(cfg: dict, params) -> torch.optim.Optimizer:
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=SGD_WEIGHT_DECAY)
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']}")


@torch.no_grad()
def _evaluate(net: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> tuple[float, float]:
    net.eval()  # BN uses running stats, dropout is off
    loss_sum = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for i in range(0, len(x), batch_size):
        logits = net(x[i:i + batch_size])
        loss_sum += nn.functional.cross_entropy(logits, y[i:i + batch_size], reduction="sum")
        correct += (logits.argmax(1) == y[i:i + batch_size]).sum()
    return loss_sum.item() / len(x), correct.item() / len(x)


def run_evaluation(cfg: dict, seed: int, n_epochs: int, split=None, device=None) -> dict:
    """Train one config and return a full record (score + per-epoch learning curve).

    `split` = (x_train, y_train, x_test, y_test) can be passed in so the data is
    loaded once per seed instead of once per evaluation.
    """
    device = device or get_device()
    x_tr, y_tr, x_te, y_te = split if split is not None else data.load_split(seed, device)

    # Same seed -> same initial weights and same batch order for every config.
    # Differences between configs then come from the HPs, not from luck.
    torch.manual_seed(seed)
    shuffle_gen = torch.Generator().manual_seed(seed)

    net = ResNet8(dropout=cfg["dropout"]).to(device)
    opt = _make_optimizer(cfg, net.parameters())
    bs = cfg["batch_size"]

    history, diverged = [], False
    t_start = time.time()
    for epoch in range(n_epochs):
        t_epoch = time.time()
        net.train()
        perm = torch.randperm(len(x_tr), generator=shuffle_gen).to(device)
        # Accumulate on the device: calling .item() every step would force the
        # CPU to wait for the GPU each batch and slow training down noticeably.
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for i in range(0, len(x_tr), bs):
            idx = perm[i:i + bs]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(yb)
            correct += (logits.argmax(1) == yb).sum()

        train_loss = loss_sum.item() / len(x_tr)
        test_loss, test_acc = _evaluate(net, x_te, y_te)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": correct.item() / len(x_tr),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "epoch_time_s": time.time() - t_epoch,
        })
        # A NaN/inf loss never recovers; stop early instead of wasting the remaining epochs.
        if not math.isfinite(train_loss):
            diverged = True
            break

    return {
        "config": cfg,
        "seed": seed,
        # Score = test accuracy after the LAST epoch, not the best epoch:
        # picking the best epoch would itself be a selection on the test set.
        "score": history[-1]["test_acc"],
        "diverged": diverged,
        "epochs_completed": len(history),
        "n_epochs": n_epochs,
        "wall_time_s": time.time() - t_start,
        "history": history,
    }


def train_and_validate(hyperparameters: dict, seed: int, n_epochs: int) -> float:
    """The interface from the assignment hint."""
    return run_evaluation(hyperparameters, seed, n_epochs)["score"]
