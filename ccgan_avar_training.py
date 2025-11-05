"""Training script for the LightningCcGANAVAR module."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from CCDM_unified.label_embedding import LabelEmbed
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config.ccgan_avar_config import CcGANAVARExperimentConfig
from dataset.geometries_dataset import GeometriesDataModule, GeometriesDataset
from models.avar_networks import load_auxiliary_builder, load_network_constructors
from models.pl_ccgan_avar import AVAROptimisationConfig, LightningCcGANAVAR
from utils.training_utils import log_config_to_wandb, set_seed

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def _flatten_images(images: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    if images.ndim != 4:
        raise ValueError("Expected images with shape (N, C, H, W)")
    n, c, h, w = images.shape
    flattened = images.reshape(n, c * h * w).astype(np.float32)
    flattened = flattened / 127.5 - 1.0
    return flattened, (c, h, w)


def _extract_train_split(dataset: GeometriesDataset, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    images, _, normalized_labels = dataset.load_train_data()
    train_images = images[indices]
    train_labels = normalized_labels[indices]
    return train_images, train_labels


def _parse_channel_multipliers(
    values: Optional[Union[Iterable[int], str]]
) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, str):
        separators = [",", "_", " "]
        tokens: List[str] = [values]
        for separator in separators:
            if separator in values:
                tokens = [tok for tok in values.replace(separator, " ").split() if tok]
                break
        return [int(token) for token in tokens]
    return [int(v) for v in values]


def _filter_kwargs(constructor, kwargs: Dict[str, object]) -> Dict[str, object]:
    signature = inspect.signature(constructor)
    valid = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in valid}


def _build_callbacks(logging_config) -> List[Callback]:
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

    callbacks: List[Callback] = [checkpoint_callback]
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


def _prepare_label_embed_kwargs(config, model_cfg, dataset: GeometriesDataset) -> Dict[str, object]:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    h_dim = config.h_dim or model_cfg.dim_y
    cov_dim = config.cov_dim or (model_cfg.img_size ** 2 * model_cfg.num_channels)
    nc = config.nc or model_cfg.num_channels
    Path(config.path_y2h).mkdir(parents=True, exist_ok=True)
    Path(config.path_y2cov).mkdir(parents=True, exist_ok=True)
    return dict(
        dataset=dataset,
        path_y2h=config.path_y2h,
        path_y2cov=config.path_y2cov,
        y2h_type=config.y2h_type,
        y2cov_type=config.y2cov_type,
        h_dim=h_dim,
        cov_dim=cov_dim,
        batch_size=config.batch_size,
        nc=nc,
        device=device,
    )


def _build_networks(model_cfg):
    gen_ctor, disc_ctor = load_network_constructors(model_cfg.net_name)
    ch_multi_g = _parse_channel_multipliers(model_cfg.ch_multi_g)
    ch_multi_d = _parse_channel_multipliers(model_cfg.ch_multi_d)

    generator_kwargs = dict(
        dim_z=model_cfg.dim_z,
        dim_y=model_cfg.dim_y,
        nc=model_cfg.num_channels,
        img_size=model_cfg.img_size,
        gene_ch=model_cfg.gene_ch,
        ch_multi=ch_multi_g,
        use_sn=model_cfg.use_sn,
        use_attn=model_cfg.use_attn,
    )
    generator = gen_ctor(**_filter_kwargs(gen_ctor, generator_kwargs))

    discriminator_kwargs = dict(
        dim_y=model_cfg.dim_y,
        nc=model_cfg.num_channels,
        img_size=model_cfg.img_size,
        disc_ch=model_cfg.disc_ch,
        ch_multi=ch_multi_d,
        use_sn=model_cfg.use_sn,
        use_attn=model_cfg.use_attn,
        use_aux_reg=model_cfg.use_aux_reg_branch,
        use_aux_dre=model_cfg.use_dre_reg,
        dre_head_arch=model_cfg.dre_head_arch,
    )
    discriminator = disc_ctor(**_filter_kwargs(disc_ctor, discriminator_kwargs))

    return generator, discriminator


def _build_aux_reg_model(model_cfg, aux_cfg):
    if not model_cfg.use_aux_reg_model:
        return None
    checkpoint_path = model_cfg.aux_reg_model_checkpoint or aux_cfg.aux_reg_checkpoint
    if checkpoint_path is None:
        raise ValueError(
            "aux_reg_model_checkpoint must be provided when use_aux_reg_model is true"
        )
    builder = load_auxiliary_builder(model_cfg.aux_reg_model_arch)
    network = builder(nc=model_cfg.num_channels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("net_state_dict", checkpoint)
    network.load_state_dict(state_dict)
    network.eval()
    network.requires_grad_(False)
    if torch.cuda.is_available():
        network = network.to("cuda")
    return network


def _build_aux_loss_params(aux_cfg, model_cfg) -> Dict[str, object]:
    return dict(
        aux_reg_loss_type=aux_cfg.aux_reg_loss_type,
        aux_reg_loss_ei_hinge_factor=aux_cfg.aux_reg_loss_ei_hinge_factor,
        aux_reg_loss_huber_delta=aux_cfg.aux_reg_loss_huber_delta,
        aux_reg_loss_huber_quantile=aux_cfg.aux_reg_loss_huber_quantile,
        weight_d_aux_reg_loss=aux_cfg.weight_d_aux_reg_loss,
        weight_g_aux_reg_loss=aux_cfg.weight_g_aux_reg_loss,
        aux_reg_checkpoint=aux_cfg.aux_reg_checkpoint,
        dre_lambda=aux_cfg.dre_lambda,
        weight_d_aux_dre_loss=aux_cfg.weight_d_aux_dre_loss,
        weight_g_aux_dre_loss=aux_cfg.weight_g_aux_dre_loss,
        dre_checkpoint=aux_cfg.dre_checkpoint,
        use_aux_reg_branch=model_cfg.use_aux_reg_branch,
        use_aux_reg_model=model_cfg.use_aux_reg_model,
        use_dre_reg=model_cfg.use_dre_reg,
    )


def train(args: argparse.Namespace, wandb_run: "Run" | None = None) -> None:
    set_seed(args.seed)

    with open(args.config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    config = CcGANAVARExperimentConfig.from_dict(config_dict)

    dataset_cfg = config.dataset
    data_module = GeometriesDataModule(
        data_dir=dataset_cfg.im_path,
        im_size=dataset_cfg.im_size,
        im_channels=dataset_cfg.im_channels,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        use_latents=dataset_cfg.use_latents,
        latent_path=dataset_cfg.latent_path,
        condition_config=dataset_cfg.condition_config,
        train_val_test_split=dataset_cfg.train_val_test_split,
        max_samples=dataset_cfg.max_samples,
    )
    data_module.setup()

    if not isinstance(data_module.dataset, GeometriesDataset):
        raise TypeError("Expected GeometriesDataset as the underlying dataset")

    train_indices = np.asarray(data_module.train_dataset.indices)
    train_images, train_labels = _extract_train_split(data_module.dataset, train_indices)
    flattened_images, sample_shape = _flatten_images(train_images)

    eval_labels = data_module.dataset.load_train_data()[2]

    label_embed_kwargs = _prepare_label_embed_kwargs(config.label_embedding, config.model, data_module.dataset)
    label_embedder = LabelEmbed(**label_embed_kwargs)

    generator, discriminator = _build_networks(config.model)
    aux_reg_model = _build_aux_reg_model(config.model, config.aux_loss)

    optimisation_config = AVAROptimisationConfig(
        generator_lr=config.optimisation.generator_lr,
        discriminator_lr=config.optimisation.discriminator_lr,
        latent_dim=config.optimisation.latent_dim,
        batch_size_disc=config.optimisation.batch_size_disc,
        batch_size_gene=config.optimisation.batch_size_gene,
        num_d_steps=config.optimisation.num_d_steps,
        num_grad_acc_d=config.optimisation.num_grad_acc_d,
        num_grad_acc_g=config.optimisation.num_grad_acc_g,
        max_grad_norm=config.optimisation.max_grad_norm,
        betas=tuple(config.optimisation.betas),
    )

    if config.training.save_images_dir:
        Path(config.training.save_images_dir).mkdir(parents=True, exist_ok=True)

    module = LightningCcGANAVAR(
        train_samples=torch.from_numpy(flattened_images),
        train_labels=torch.from_numpy(train_labels.astype(np.float32)),
        eval_labels=torch.from_numpy(eval_labels.astype(np.float32)),
        generator=generator,
        discriminator=discriminator,
        fn_y2h=label_embedder.fn_y2h,
        vicinal_params=dict(vars(config.vicinal)),
        aux_loss_params=_build_aux_loss_params(config.aux_loss, config.model),
        optimisation_config=optimisation_config,
        sample_shape=sample_shape,
        loss_type=config.training.loss_type,
        use_diffaug=config.training.diffaug.enabled,
        diffaug_policy=config.training.diffaug.policy,
        use_amp=config.training.use_amp,
        mixed_precision_type=config.training.mixed_precision_type,
        use_ema=config.training.ema.enabled,
        ema_update_after_step=config.training.ema.update_after_step,
        ema_update_every=config.training.ema.update_every,
        ema_decay=config.training.ema.decay,
        sample_every_n_steps=config.training.sample_every_n_steps,
        sample_batch_size=config.training.sample_batch_size,
        save_images_folder=config.training.save_images_dir,
        checkpoint_sampling_freq=config.training.checkpoint_sampling_freq,
        geometry_label_schema=config.dataset.geometry_label_schema,
        aux_reg_model=aux_reg_model,
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
        enable_checkpointing=True,
        limit_val_batches=0,
    )

    log_config_to_wandb(logger, config_dict, args.seed)

    trainer.fit(module, train_dataloaders=data_module.train_dataloader())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightningCcGANAVAR module")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="./config/ccgan_avar_geometries.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

