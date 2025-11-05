"""Training entry point for the conditional consistency diffusion model (CCDM).

The script mirrors the ergonomics of :mod:`joint_training.py` and
:mod:`ccgan_training.py` by loading a YAML configuration, preparing the geometry
DataModule, instantiating :class:`models.LightningCCDM` and configuring
Weights & Biases logging alongside checkpointing callbacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import MISSING, fields
from typing import Any, Dict, Iterable, Callable, Optional, TYPE_CHECKING

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset.geometries_dataset import GeometriesDataModule, GeometriesDataset
from models import LightningCCDM
from models.pl_ccdm import VicinalParams
from utils.training_utils import log_config_to_wandb, set_seed

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def _as_tuple(values: Iterable[float]) -> tuple:
    """Convert ``values`` to a tuple, preserving ``None``.

    The helper keeps configuration parsing compact whilst ensuring that values
    destined for Lightning (such as ``train_val_test_split``) use concrete
    tuples.
    """

    if values is None:
        return values
    if isinstance(values, tuple):
        return values
    return tuple(values)


def _build_data_module(dataset_config: Dict[str, Any]) -> GeometriesDataModule:
    """Initialise the :class:`GeometriesDataModule` from configuration."""

    condition_config = dataset_config.get("condition_config")
    split = _as_tuple(dataset_config.get("train_val_test_split", (0.7, 0.15, 0.15)))

    return GeometriesDataModule(
        data_dir=dataset_config["im_path"],
        im_size=dataset_config.get("im_size", 128),
        im_channels=dataset_config.get("im_channels", 1),
        batch_size=dataset_config.get("batch_size", 32),
        num_workers=dataset_config.get("num_workers", 4),
        use_latents=dataset_config.get("use_latents", False),
        latent_path=dataset_config.get("latent_path"),
        condition_config=condition_config,
        train_val_test_split=split,
        max_samples=dataset_config.get("max_samples"),
    )


def _build_callbacks(logging_config: Dict[str, Any]) -> list:
    """Create the list of callbacks for Lightning."""

    checkpoint_dir = logging_config.get("checkpoint_dir", "./checkpoints/ccdm")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=logging_config.get("checkpoint_filename", "ccdm-{step}"),
        monitor=logging_config.get("checkpoint_monitor"),
        mode=logging_config.get("checkpoint_mode", "min"),
        save_top_k=logging_config.get("save_top_k", 1),
        save_last=True,
        every_n_train_steps=logging_config.get("checkpoint_every_n_steps"),
        every_n_epochs=logging_config.get("checkpoint_every_n_epochs"),
    )

    callbacks: list = [checkpoint_callback]

    if logging_config.get("enable_lr_monitor", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return callbacks


def _build_logger(logging_config: Dict[str, Any], wandb_run: "Run" | None = None) -> WandbLogger:
    """Initialise the Weights & Biases logger."""

    project = logging_config.get("wandb_project", "CCDMGeometry")
    entity = logging_config.get("wandb_entity")

    return WandbLogger(
        project=project,
        entity=entity,
        log_model=logging_config.get("log_model", True),
        experiment=wandb_run,
    )


def _instantiate_module(config: Dict[str, Any]) -> LightningCCDM:
    """Create the :class:`LightningCCDM` module from configuration."""

    dataset_config = config.get("dataset_params", {})
    ccdm_config = dict(config.get("ccdm_params", {}))
    unet_config = dict(config.get("unet_params", {}))
    diffusion_config = dict(config.get("diffusion_params", {}))
    label_embed_config = dict(config.get("label_embedding_params", {}))
    vicinal_config = dict(config.get("vicinal_params", {}))
    sampling_config = dict(config.get("sampling_params", {}))

    def _to_tuple(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    image_size = (
        ccdm_config.get("image_size")
        or dataset_config.get("im_size")
        or unet_config.get("image_size")
    )
    if image_size is None:
        raise KeyError("Configuration must define an image size")

    in_channels = (
        ccdm_config.get("in_channels")
        or dataset_config.get("im_channels")
        or unet_config.get("in_channels")
    )
    if in_channels is None:
        raise KeyError("Configuration must define the number of image channels")

    lr = float(ccdm_config.get("train_lr", 1e-4))
    adam_betas_values = ccdm_config.get("adam_betas", (0.9, 0.99))
    if isinstance(adam_betas_values, (list, tuple)):
        adam_betas = tuple(float(beta) for beta in adam_betas_values)
        if len(adam_betas) != 2:
            raise ValueError("adam_betas must contain exactly two values")
    else:
        raise TypeError("adam_betas must be a sequence of two floats in the configuration")
    ema_decay = float(ccdm_config.get("ema_decay", 0.995))
    ema_update_every = int(ccdm_config.get("ema_update_every", 10))
    ema_update_after_step = int(ccdm_config.get("ema_update_after_step", 100))
    results_folder = ccdm_config.get("results_folder")

    sample_every_n_steps = int(
        sampling_config.get(
            "log_every_n_training_steps",
            ccdm_config.get("sample_every", 0),
        )
    )
    sample_every_n_epochs = int(
        sampling_config.get(
            "log_every_n_epochs",
            1,
        )
    )
    sample_batch_size = int(
        sampling_config.get(
            "sample_batch_size",
            ccdm_config.get(
                "train_batch_size",
                dataset_config.get("batch_size", 64),
            ),
        )
    )
    cond_scale = float(
        sampling_config.get(
            "guidance_scale",
            ccdm_config.get("cond_scale_visual", 1.5),
        )
    )

    use_y_covariance = bool(diffusion_config.pop("use_Hy", False))

    if "num_inference_steps" in sampling_config and "sample_timesteps" not in diffusion_config:
        diffusion_config.setdefault(
            "sample_timesteps", sampling_config["num_inference_steps"]
        )

    unet_key_map = {
        "model_channels": "dim",
        "channel_mults": "dim_mults",
    }
    allowed_unet_keys = {
        "dim",
        "embed_input_dim",
        "cond_drop_prob",
        "init_dim",
        "out_dim",
        "dim_mults",
        "in_channels",
        "learned_variance",
        "learned_sinusoidal_cond",
        "random_fourier_features",
        "learned_sinusoidal_dim",
        "attn_dim_head",
        "attn_heads",
    }
    unet_kwargs: Dict[str, Any] = {}
    for key, value in unet_config.items():
        mapped_key = unet_key_map.get(key, key)
        if mapped_key in allowed_unet_keys:
            if mapped_key == "dim_mults":
                value = _to_tuple(value)
            unet_kwargs[mapped_key] = value

    diffusion_key_map = {
        "train_timesteps": "timesteps",
        "sample_timesteps": "sampling_timesteps",
        "pred_objective": "objective",
        "ddim_eta": "ddim_sampling_eta",
    }
    allowed_diffusion_keys = {
        "image_size",
        "use_Hy",
        "fn_y2cov",
        "cond_drop_prob",
        "timesteps",
        "sampling_timesteps",
        "objective",
        "beta_schedule",
        "ddim_sampling_eta",
        "offset_noise_strength",
        "min_snr_loss_weight",
        "min_snr_gamma",
        "use_cfg_plus_plus",
    }
    diffusion_kwargs: Dict[str, Any] = {}
    for key, value in diffusion_config.items():
        mapped_key = diffusion_key_map.get(key, key)
        if mapped_key in {"timesteps", "sampling_timesteps"}:
            value = int(value)
        if mapped_key in allowed_diffusion_keys:
            diffusion_kwargs[mapped_key] = value

    deferred_covariance: Optional["_DeferredCovariance"] = None

    if use_y_covariance and "fn_y2cov" not in diffusion_kwargs:

        class _DeferredCovariance:
            """Lazy proxy for the label covariance embedding."""

            def __init__(self) -> None:
                self._state: Dict[str, Optional[Callable[..., Any]]] = {"fn": None}

            def initialise(self, fn: Callable[..., Any]) -> None:
                self._state["fn"] = fn

            def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - runtime guard
                fn = self._state["fn"]
                if fn is None:
                    raise RuntimeError("Label covariance function accessed before initialisation")
                return fn(*args, **kwargs)

            def __deepcopy__(self, memo: Dict[int, Any]) -> "_DeferredCovariance":  # pragma: no cover - deepcopy support
                new = type(self).__new__(type(self))
                memo[id(self)] = new
                new._state = self._state
                return new

        deferred_covariance = _DeferredCovariance()
        diffusion_kwargs["fn_y2cov"] = deferred_covariance

    label_embed_key_map = {
        "dim_embed": "h_dim",
        "y2h_embed_type": "y2h_type",
        "y2cov_embed_type": "y2cov_type",
    }
    allowed_label_embed_keys = {
        "dataset",
        "path_y2h",
        "path_y2cov",
        "y2h_type",
        "y2cov_type",
        "h_dim",
        "cov_dim",
        "batch_size",
        "nc",
        "device",
    }
    label_embed_kwargs: Dict[str, Any] = {}
    for key, value in label_embed_config.items():
        mapped_key = label_embed_key_map.get(key, key)
        if mapped_key in allowed_label_embed_keys:
            label_embed_kwargs[mapped_key] = value

    requires_resnet_embeddings = any(
        label_embed_kwargs.get(key) == "resnet" for key in ("y2h_type", "y2cov_type")
    )
    if requires_resnet_embeddings and "dataset" not in label_embed_kwargs:
        dataset_kwargs = dict(
            im_path=dataset_config["im_path"],
            im_size=int(dataset_config.get("im_size", image_size)),
            im_channels=int(dataset_config.get("im_channels", in_channels)),
            use_latents=dataset_config.get("use_latents", False),
            latent_path=dataset_config.get("latent_path"),
            condition_config=dataset_config.get("condition_config"),
            max_samples=dataset_config.get("max_samples"),
        )
        label_embed_kwargs.setdefault("dataset", GeometriesDataset(**dataset_kwargs))

    vicinal_defaults: Dict[str, Any] = {}
    for field in fields(VicinalParams):
        if field.default is not MISSING:
            vicinal_defaults[field.name] = field.default
    vicinal_defaults.setdefault("kernel_sigma", 0.0)
    vicinal_defaults.setdefault("kappa", 0.0)
    for field in fields(VicinalParams):
        if field.name in vicinal_config:
            vicinal_defaults[field.name] = vicinal_config[field.name]
    vicinal_params = VicinalParams(**vicinal_defaults)

    lightning_module = LightningCCDM(
        image_size=int(image_size),
        in_channels=int(in_channels),
        vicinal_params=vicinal_params,
        unet_kwargs=unet_kwargs or None,
        diffusion_kwargs=diffusion_kwargs or None,
        label_embed_kwargs=label_embed_kwargs or None,
        use_y_covariance=use_y_covariance,
        lr=lr,
        adam_betas=adam_betas,
        ema_decay=ema_decay,
        ema_update_every=ema_update_every,
        ema_update_after_step=ema_update_after_step,
        sample_every_n_steps=sample_every_n_steps,
        sample_every_n_epochs=sample_every_n_epochs,
        results_folder=results_folder,
        sample_batch_size=sample_batch_size,
        cond_scale=cond_scale,
    )
    if use_y_covariance:
        fn_y2cov = lightning_module.label_embed.fn_y2cov
        if deferred_covariance is not None:
            deferred_covariance.initialise(fn_y2cov)
        lightning_module.diffusion.fn_y2cov = fn_y2cov
        if hasattr(lightning_module, "ema") and hasattr(lightning_module.ema, "ema_model"):
            lightning_module.ema.ema_model.fn_y2cov = fn_y2cov
    return lightning_module


def train(args: argparse.Namespace, wandb_run: "Run" | None = None) -> None:
    """Execute the CCDM training pipeline."""

    set_seed(args.seed)

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config: Dict[str, Any] = yaml.safe_load(config_file)

    dataset_config = config["dataset_params"]
    logging_config = config.get("logging_params", {})
    trainer_config = config.get("trainer_params", {})

    data_module = _build_data_module(dataset_config)
    lightning_module = _instantiate_module(config)

    wandb_logger = _build_logger(logging_config, wandb_run=wandb_run)
    log_config_to_wandb(wandb_logger, config, args.seed)

    callbacks = _build_callbacks(logging_config)

    precision = trainer_config.get("precision", "32-true")
    accelerator = trainer_config.get("accelerator")
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=trainer_config.get("devices", 1),
        logger=wandb_logger,
        callbacks=callbacks,
        max_steps=trainer_config.get("max_steps"),
        max_epochs=trainer_config.get("max_epochs"),
        gradient_clip_val=trainer_config.get("gradient_clip_val"),
        accumulate_grad_batches=trainer_config.get("accumulate_grad_batches", 1),
        precision=precision,
        log_every_n_steps=logging_config.get("log_every_n_steps", 50),
        default_root_dir=logging_config.get("default_root_dir"),
        check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch", 1),
        enable_checkpointing=True,
    )

    trainer.fit(lightning_module, datamodule=data_module)


def parse_args() -> argparse.Namespace:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="Train a CCDM Lightning module")
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        default="./config/ccdm_geometries.yaml",
        help="Path to the CCDM configuration file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
