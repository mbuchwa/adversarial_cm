"""Training script for the LightningCcGAN module."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config.ccgan_config import CcGANExperimentConfig
from dataset.geometries_dataset import GeometriesDataModule, GeometriesDataset
from models.ccgan_networks import (
    ContinuousConditionalDiscriminator,
    ContinuousConditionalGenerator,
    ContinuousDiscriminatorConfig,
    ContinuousGeneratorConfig,
)
from models.pl_ccgan import LightningCcGAN, OptimisationConfig, VicinalSamplingConfig
from utils.training_utils import log_config_to_wandb, set_seed

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def _flatten_images(images: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, C, H, W)")
    n, c, h, w = images.shape
    flattened = images.reshape(n, c * h * w)
    return flattened.astype(np.float32) / 255.0, (c, h, w)


def _extract_train_split(dataset: GeometriesDataset, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    images, raw_labels, normalized_labels = dataset.load_train_data()
    train_images = images[indices]
    train_labels = normalized_labels[indices]
    return train_images, train_labels


def _build_callbacks(logging_config) -> list:
    checkpoint_dir = Path(logging_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=logging_config.checkpoint_filename,
        monitor=logging_config.checkpoint_monitor,
        mode=logging_config.checkpoint_mode,
        save_top_k=logging_config.save_top_k,
        save_last=True,
    )

    callbacks = [checkpoint_callback]
    if logging_config.enable_lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def _build_logger(logging_config, wandb_run: "Run" | None = None) -> WandbLogger:
    return WandbLogger(
        project=logging_config.wandb_project,
        entity=logging_config.wandb_entity,
        log_model=logging_config.log_model,
        experiment=wandb_run,
    )


def train(args: argparse.Namespace, wandb_run: "Run" | None = None) -> None:
    set_seed(args.seed)

    with open(args.config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    config = CcGANExperimentConfig.from_dict(config_dict)

    data_module = GeometriesDataModule(
        data_dir=config.dataset.im_path,
        im_size=config.dataset.im_size,
        im_channels=config.dataset.im_channels,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        use_latents=config.dataset.use_latents,
        latent_path=config.dataset.latent_path,
        condition_config=config.dataset.condition_config,
        train_val_test_split=config.dataset.train_val_test_split,
        max_samples=config.dataset.max_samples,
    )
    data_module.setup()

    if not isinstance(data_module.train_dataset.dataset, GeometriesDataset):
        raise TypeError("Expected GeometriesDataset for the training split")

    train_indices = np.asarray(data_module.train_dataset.indices)
    train_images, train_labels = _extract_train_split(
        data_module.train_dataset.dataset, train_indices
    )

    flattened_images, sample_shape = _flatten_images(train_images)
    label_dim = train_labels.shape[1] if train_labels.ndim > 1 else 1

    generator_cfg = ContinuousGeneratorConfig(
        latent_dim=config.model.latent_dim,
        output_dim=flattened_images.shape[1],
        hidden_dim=config.model.hidden_dim,
        num_hidden_layers=config.model.generator_layers,
        label_dim=label_dim,
        radius=config.model.radius,
    )
    discriminator_cfg = ContinuousDiscriminatorConfig(
        input_dim=flattened_images.shape[1],
        hidden_dim=config.model.hidden_dim,
        num_hidden_layers=config.model.discriminator_layers,
        label_dim=label_dim,
        radius=config.model.radius,
    )

    generator = ContinuousConditionalGenerator(generator_cfg)
    discriminator = ContinuousConditionalDiscriminator(discriminator_cfg)

    vicinal_config = VicinalSamplingConfig(
        kernel_sigma=config.vicinal.kernel_sigma,
        kappa=config.vicinal.kappa,
        threshold_type=config.vicinal.threshold_type,
        nonzero_soft_weight_threshold=config.vicinal.nonzero_soft_weight_threshold,
        circular_labels=config.vicinal.circular_labels,
    )

    optimisation_config = OptimisationConfig(
        generator_lr=config.optimisation.generator_lr,
        discriminator_lr=config.optimisation.discriminator_lr,
        batch_size_disc=config.optimisation.batch_size_disc,
        batch_size_gene=config.optimisation.batch_size_gene,
        latent_dim=config.model.latent_dim,
    )

    module = LightningCcGAN(
        train_samples=torch.from_numpy(flattened_images),
        train_labels=torch.from_numpy(train_labels.astype(np.float32)),
        generator=generator,
        discriminator=discriminator,
        vicinal_config=vicinal_config,
        optimisation_config=optimisation_config,
        sample_shape=sample_shape,
        sample_every_n_steps=config.training.sample_every_n_steps,
        sample_batch_size=config.training.sample_batch_size,
    )

    logger = _build_logger(config.logging, wandb_run=wandb_run)
    callbacks = _build_callbacks(config.logging)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=config.training.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    log_config_to_wandb(logger, config_dict, args.seed)

    trainer.fit(module, train_dataloaders=data_module.train_dataloader())


def parse_args() -> argparse.Namespace:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="Train the LightningCcGAN module")
    parser.add_argument("--config",
                        dest="config_path",
                        type=str,
                        default="./config/ccgan_geometries.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

