"""Search space: the grid (96 configs) and the random-search distributions.

A config is a plain dict. Conditional HPs are only present when active:
  SGD   -> "momentum"      (weight decay fixed to SGD_WEIGHT_DECAY, not searched)
  AdamW -> "weight_decay"
"""

import itertools

import numpy as np

BATCH_SIZES = [64, 128, 256]
OPTIMIZERS = ["sgd", "adamw"]
SGD_WEIGHT_DECAY = 5e-4  # fixed, as in He et al. (2016)

# --- Grid: 3 x 4 x 2 x (2 + 2) = 96 ---
GRID_LR = [1e-4, 1e-3, 1e-2, 1e-1]
GRID_DROPOUT = [0.0, 0.3]
GRID_CONDITIONAL = {
    "sgd": ("momentum", [0.5, 0.9]),
    "adamw": ("weight_decay", [1e-4, 1e-2]),
}

# --- Random search ranges ---
LR_RANGE = (1e-4, 1e-1)              # log-uniform
DROPOUT_RANGE = (0.0, 0.5)           # uniform
MOMENTUM_RANGE = (0.5, 0.99)         # uniform
WEIGHT_DECAY_RANGE = (1e-5, 1e-1)    # log-uniform


def grid_configs() -> list[dict]:
    configs = []
    for opt in OPTIMIZERS:
        cond_name, cond_values = GRID_CONDITIONAL[opt]
        # The conditional HP only multiplies its own optimizer's branch,
        # which is why the grid size is a sum over optimizers, not a product.
        for bs, lr, do, cv in itertools.product(BATCH_SIZES, GRID_LR, GRID_DROPOUT, cond_values):
            configs.append({"batch_size": bs, "lr": lr, "dropout": do, "optimizer": opt, cond_name: cv})
    return configs


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    # Uniform in log-space: each order of magnitude gets the same probability.
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def random_configs(n: int, seed: int) -> list[dict]:
    """All n configs are drawn upfront from `seed`, so a resumed run gets the exact same list."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        cfg = {
            "batch_size": int(rng.choice(BATCH_SIZES)),
            "lr": _log_uniform(rng, *LR_RANGE),
            "dropout": float(rng.uniform(*DROPOUT_RANGE)),
            "optimizer": str(rng.choice(OPTIMIZERS)),
        }
        # Conditional HP: sampled only after the optimizer is known.
        if cfg["optimizer"] == "sgd":
            cfg["momentum"] = float(rng.uniform(*MOMENTUM_RANGE))
        else:
            cfg["weight_decay"] = _log_uniform(rng, *WEIGHT_DECAY_RANGE)
        configs.append(cfg)
    return configs


if __name__ == "__main__":
    grid = grid_configs()
    print(f"grid size: {len(grid)}")
    for c in grid[:3] + grid[-3:]:
        print("  ", c)
    print("random (seed 42):")
    for c in random_configs(4, 42):
        print("  ", c)
