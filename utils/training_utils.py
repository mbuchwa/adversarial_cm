"""Common helpers for training scripts."""

import random
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for deterministic behaviour."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


@rank_zero_only
def log_config_to_wandb(wandb_logger: WandbLogger, config: Dict, seed: int) -> None:
    """Record the loaded configuration and seed to Weights & Biases."""

    wandb_logger.experiment.config.update(config)
    wandb_logger.experiment.config.update({"random_seed": seed})

