"""PyTorch Lightning module wrapping the CCDM trainer logic.

The goal of :class:`LightningCCDM` is to faithfully reproduce the training
behaviour of :mod:`CCDM_unified.trainer.Trainer` while exposing a Lightning
compatible API.  The implementation mirrors the original optimisation loop,
including vicinal sampling, EMA tracking and periodic sample generation.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

try:  # pragma: no cover - optional dependency during tests
    import wandb
except ModuleNotFoundError:  # pragma: no cover - wandb optional at runtime
    wandb = None  # type: ignore

from CCDM_unified.diffusion import GaussianDiffusion
from CCDM_unified.ema_pytorch import EMA
from CCDM_unified.label_embedding import LabelEmbed
from CCDM_unified.models import Unet
from CCDM_unified.utils import normalize_images
from utils.evaluation_utils import ImageSimilarityMetrics


class _LightningModelWrapper(nn.Module):
    """Minimal wrapper that mimics :class:`nn.DataParallel`.

    The original diffusion implementation expects a ``module`` attribute on the
    model that is passed in.  Lightning modules typically operate on the raw
    module, so this wrapper simply forwards calls while exposing the wrapped
    module through ``.module``.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover -
        return self.module(*args, **kwargs)


@dataclass
class VicinalParams:
    """Configuration for vicinal sampling."""

    kernel_sigma: float
    kappa: float
    threshold_type: str = "hard"
    nonzero_soft_weight_threshold: float = 0.0
    distance_metric: str = "euclidean"
    kernel: str = "gaussian"


class LightningCCDM(pl.LightningModule):
    """Lightning reimplementation of the CCDM trainer.

    Parameters mirror the ones in :mod:`CCDM_unified.main`.  Instead of relying
    on Accelerate this module uses Lightning's training loop while keeping the
    core sampling and optimisation logic unchanged.  Visual samples can be
    logged periodically every ``sample_every_n_steps`` training steps and, if
    desired, additionally every ``sample_every_n_epochs`` epochs.
    """

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        train_images: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        vicinal_params: VicinalParams = VicinalParams(
            kernel_sigma=0.0, kappa=0.0, threshold_type="hard"
        ),
        unet_kwargs: Optional[Mapping[str, Any]] = None,
        diffusion_kwargs: Optional[Mapping[str, Any]] = None,
        label_embed_kwargs: Optional[Mapping[str, Any]] = None,
        use_y_covariance: bool = False,
        lr: float = 1e-4,
        adam_betas: Tuple[float, float] = (0.9, 0.99),
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        ema_update_after_step: int = 100,
        sample_every_n_steps: int = 1000,
        sample_every_n_epochs: int = 1,
        results_folder: Optional[str] = None,
        sample_batch_size: int = 64,
        cond_scale: float = 1.5,
    ) -> None:
        super().__init__()

        if train_images is not None and train_labels is None:
            raise ValueError("train_labels must be provided when train_images are supplied")

        self.save_hyperparameters(
            ignore=[
                "train_images",
                "train_labels",
                "unet_kwargs",
                "diffusion_kwargs",
                "label_embed_kwargs",
            ]
        )

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.lr = float(lr)
        self.adam_betas = adam_betas
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.sample_every_n_epochs = int(sample_every_n_epochs)
        self.sample_batch_size = int(sample_batch_size)
        self.cond_scale = float(cond_scale)
        self._last_sample_step = -1

        self.vicinal_params = vicinal_params
        self.kernel_sigma = float(vicinal_params.kernel_sigma)
        self.kappa = float(vicinal_params.kappa)
        self.threshold_type = vicinal_params.threshold_type
        self.nonzero_soft_weight_threshold = float(
            vicinal_params.nonzero_soft_weight_threshold
        )
        self.distance_metric = vicinal_params.distance_metric
        self.kernel = vicinal_params.kernel

        if self.threshold_type not in {"hard", "soft"}:
            raise ValueError("threshold_type must be either 'hard' or 'soft'")
        if self.distance_metric not in {"euclidean", "mahalanobis"}:
            raise ValueError("Unsupported distance metric for vicinal sampling")
        if self.kernel != "gaussian":
            raise ValueError("Only gaussian vicinal kernels are currently supported")

        self.soft_threshold: Optional[float] = None
        if self.threshold_type == "soft":
            if self.nonzero_soft_weight_threshold <= 0 or self.kappa <= 0:
                raise ValueError(
                    "Soft vicinal weighting requires positive kappa and non-zero threshold"
                )
            self.soft_threshold = (
                -math.log(self.nonzero_soft_weight_threshold) / self.kappa
            )

        # ------------------------------------------------------------------
        # Model components
        # ------------------------------------------------------------------
        unet_kwargs = dict(unet_kwargs or {})
        default_unet_kwargs = dict(
            dim=128,
            embed_input_dim=128,
            cond_drop_prob=0.0,
            dim_mults=(1, 2, 4, 8),
            in_channels=self.in_channels,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            attn_dim_head=32,
            attn_heads=4,
        )
        for key, value in default_unet_kwargs.items():
            unet_kwargs.setdefault(key, value)

        self.unet = Unet(**unet_kwargs)
        self.model = _LightningModelWrapper(self.unet)

        diffusion_kwargs = dict(diffusion_kwargs or {})
        default_diffusion_kwargs = dict(
            image_size=self.image_size,
            use_Hy=use_y_covariance,
            cond_drop_prob=unet_kwargs.get("cond_drop_prob", 0.0),
            timesteps=1000,
            sampling_timesteps=None,
            objective="pred_noise",
            beta_schedule="cosine",
            ddim_sampling_eta=0.0,
        )
        for key, value in default_diffusion_kwargs.items():
            diffusion_kwargs.setdefault(key, value)

        self.diffusion = GaussianDiffusion(self.model, **diffusion_kwargs)
        self.channels = self.diffusion.channels

        label_embed_kwargs = dict(label_embed_kwargs or {})
        default_label_embed_kwargs = dict(
            dataset=None,
            path_y2h=os.path.join(results_folder or "./results", "model_y2h"),
            path_y2cov=os.path.join(results_folder or "./results", "model_y2cov"),
            y2h_type="sinusoidal",
            y2cov_type="sinusoidal",
            h_dim=unet_kwargs.get("embed_input_dim", 128),
            cov_dim=self.image_size ** 2 * self.in_channels,
            batch_size=128,
            nc=self.in_channels,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        for key, value in default_label_embed_kwargs.items():
            label_embed_kwargs.setdefault(key, value)

        y2h_type = label_embed_kwargs.get("y2h_type")
        y2cov_type = label_embed_kwargs.get("y2cov_type")
        if (
            (y2h_type == "resnet" or y2cov_type == "resnet")
            and label_embed_kwargs.get("dataset") is None
        ):
            raise ValueError(
                "ResNet label embeddings require label_embed_kwargs['dataset'] to provide a "
                "dataset implementing load_train_data(), or preloaded arrays must be supplied."
            )

        self.label_embed = LabelEmbed(**label_embed_kwargs)
        self.fn_y2h = self.label_embed.fn_y2h
        self.fn_y2cov = self.label_embed.fn_y2cov if diffusion_kwargs.get("use_Hy", False) else None

        self.ema_decay = float(ema_decay)
        self.ema_update_every = int(ema_update_every)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema = EMA(
            self.diffusion,
            beta=self.ema_decay,
            update_every=self.ema_update_every,
            update_after_step=self.ema_update_after_step,
        )

        # ------------------------------------------------------------------
        # Dataset caches (filled during ``setup``)
        # ------------------------------------------------------------------
        self._train_images_cpu: Optional[Tensor] = None
        self._train_labels_cpu: Optional[Tensor] = None
        self._conditioning_dim: Optional[int] = None
        self._label_cov_inv: Optional[Tensor] = None
        self._label_mean: Optional[Tensor] = None
        self._logged_conditioning_metadata = False

        if train_images is not None and train_labels is not None:
            self._cache_external_dataset(train_images, train_labels)

        self.results_folder = Path(results_folder or "./results")
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self._visual_labels: Optional[Tensor] = None
        self._visual_grid_size = max(1, int(math.sqrt(self.sample_batch_size)))

        metrics_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_calculator = ImageSimilarityMetrics(device=metrics_device)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        super().setup(stage)
        self.ema.to(self.device)
        if self.trainer is not None and self._train_images_cpu is None:
            self._populate_dataset_cache_from_datamodule()

        if self._visual_labels is None and self._train_labels_cpu is not None:
            self._visual_labels = self._build_visual_label_grid()

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = Adam(self.diffusion.parameters(), lr=self.lr, betas=self.adam_betas)
        return optimizer

    def optimizer_step(  # type: ignore[override]
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Optional[Callable[[], None]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema.update()

        if (
            self.sample_every_n_steps > 0
            and self.trainer is not None
            and self.global_step % self.sample_every_n_steps == 0
            and self.global_step > 0
        ):
            self._log_sample_grid(self.global_step)
            self._last_sample_step = self.global_step

    def training_step(self, batch: Tuple[Tensor, Mapping[str, Tensor]], batch_idx: int):
        images, cond_inputs = batch
        if "tensor" not in cond_inputs:
            raise KeyError("cond_inputs must contain a 'tensor' entry with geometry labels")

        images = images.to(self.device, dtype=torch.float)
        batch_labels_input = cond_inputs["tensor"].to(self.device).float()
        batch_labels_input = self._ensure_label_2d_tensor(batch_labels_input)
        batch_size = batch_labels_input.shape[0]

        if not self._logged_conditioning_metadata:
            self._log_conditioning_metadata(batch_labels_input)

        if self.threshold_type == "hard" and self.kappa == 0 and self.kernel_sigma == 0:
            batch_images = images
            batch_labels = batch_labels_input
            vicinal_weights = None
        else:
            batch_images, batch_labels, vicinal_weights = self._sample_training_batch(
                batch_size=batch_size, device=images.device
            )

        labels_emb = self.fn_y2h(batch_labels)
        loss = self.diffusion(
            batch_images,
            labels_emb=labels_emb,
            labels=batch_labels,
            vicinal_weights=vicinal_weights,
            current_epoch=self.current_epoch
        )

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        if vicinal_weights is not None:
            self.log(
                "train/vicinal_weight_mean",
                vicinal_weights.mean(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def validation_step(self, batch: Tuple[Tensor, Mapping[str, Tensor]], batch_idx: int):
        images, cond_inputs = batch
        if "tensor" not in cond_inputs:
            raise KeyError("cond_inputs must contain a 'tensor' entry with geometry labels")

        images = images.to(self.device, dtype=torch.float)
        labels = cond_inputs["tensor"].to(self.device).float()
        labels = self._ensure_label_2d_tensor(labels)

        labels_emb = self.fn_y2h(labels)
        loss = self.diffusion(
            images,
            labels_emb=labels_emb,
            labels=labels,
            vicinal_weights=None,
        )

        self.log("val/loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self) -> None:  # type: ignore[override]
        if self.trainer is None or self._visual_labels is None:
            return

        current_step = self.global_step
        if current_step <= 0:
            return

        step_interval_hit = (
            self.sample_every_n_steps > 0
            and current_step % self.sample_every_n_steps == 0
        )
        if step_interval_hit and self._last_sample_step == current_step:
            return

        if self.sample_every_n_epochs <= 0:
            return

        if (self.current_epoch + 1) % self.sample_every_n_epochs != 0:
            return

        self._log_sample_grid(current_step)
        self._last_sample_step = current_step

    # ------------------------------------------------------------------
    # Dataset preparation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_label_2d_tensor(labels: Tensor) -> Tensor:
        if labels.ndim == 1:
            return labels.unsqueeze(-1)
        return labels.reshape(labels.shape[0], -1)

    def _log_conditioning_metadata(self, batch_labels: Tensor) -> None:
        cond_dim = batch_labels.shape[1]
        self.log(
            "train/cond_dim",
            float(cond_dim),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self._logged_conditioning_metadata = True

    def _update_label_statistics(self) -> None:
        if self._train_labels_cpu is None:
            self._conditioning_dim = None
            self._label_cov_inv = None
            self._label_mean = None
            return

        labels = self._ensure_label_2d_tensor(self._train_labels_cpu)
        self._train_labels_cpu = labels
        self._conditioning_dim = labels.shape[1]
        self._label_mean = labels.mean(dim=0)

        if self.distance_metric == "mahalanobis":
            if labels.shape[0] > 1:
                centered = labels - self._label_mean
                cov = torch.matmul(centered.T, centered) / max(labels.shape[0] - 1, 1)
            else:
                cov = torch.eye(labels.shape[1], dtype=labels.dtype)
            cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype) * 1e-6
            self._label_cov_inv = torch.linalg.pinv(cov)
        else:
            self._label_cov_inv = None

    def _cache_external_dataset(self, images: np.ndarray, labels: np.ndarray) -> None:
        images = np.asarray(images)
        labels = np.asarray(labels)

        if images.ndim != 4:
            raise ValueError("train_images must have shape (N, C, H, W)")
        if labels.ndim not in (1, 2):
            raise ValueError("train_labels must be a 1D or 2D array")

        images_tensor = torch.from_numpy(images).float()
        if images_tensor.max() > 1.0:
            images_tensor = torch.from_numpy(
                normalize_images(images, to_neg_one_to_one=False)
            ).float()
        labels_tensor = torch.from_numpy(labels).float()
        if labels_tensor.ndim == 2 and labels_tensor.shape[1] == 1:
            labels_tensor = labels_tensor.view(-1)

        self._train_images_cpu = images_tensor
        self._train_labels_cpu = self._ensure_label_2d_tensor(labels_tensor)
        self._update_label_statistics()

    def _populate_dataset_cache_from_datamodule(self) -> None:
        datamodule = self.trainer.datamodule if self.trainer is not None else None
        if datamodule is None or not hasattr(datamodule, "train_dataset"):
            return

        train_dataset = datamodule.train_dataset  # type: ignore[attr-defined]
        if train_dataset is None:
            return

        loader = DataLoader(
            train_dataset,
            batch_size=getattr(datamodule, "batch_size", 64),
            num_workers=getattr(datamodule, "num_workers", 0),
            shuffle=False,
        )

        images: list[Tensor] = []
        labels: list[Tensor] = []
        for batch in loader:
            batch_images, cond_inputs = batch
            if "tensor" not in cond_inputs:
                raise KeyError(
                    "cond_inputs produced by GeometriesDataModule must contain 'tensor'"
                )
            images.append(batch_images.detach().cpu())
            labels.append(
                self._ensure_label_2d_tensor(cond_inputs["tensor"].detach().float()).cpu()
            )

        self._train_images_cpu = torch.cat(images, dim=0)
        self._train_labels_cpu = torch.cat(labels, dim=0)
        self._update_label_statistics()

    # ------------------------------------------------------------------
    # Sampling and logging utilities
    # ------------------------------------------------------------------
    def _build_visual_label_grid(self) -> Optional[Tensor]:
        if self._train_labels_cpu is None or len(self._train_labels_cpu) == 0:
            return None

        labels = self._train_labels_cpu
        n_row = self._visual_grid_size
        n_col = n_row
        total = n_row * n_col

        if labels.shape[1] == 1:
            labels_sorted, _ = torch.sort(labels[:, 0])
            start_label = torch.quantile(labels_sorted, 0.05)
            end_label = torch.quantile(labels_sorted, 0.95)
            grid = torch.linspace(start_label, end_label, steps=n_row)
            y_visual = torch.zeros(total, 1)
            for i in range(n_row):
                y_visual[i * n_col : (i + 1) * n_col, 0] = grid[i]
            return y_visual

        ordered = labels[torch.argsort(labels[:, 0])]
        if ordered.shape[0] >= total:
            indices = torch.linspace(0, ordered.shape[0] - 1, steps=total).long()
            return ordered[indices]

        repeats = math.ceil(total / ordered.shape[0])
        tiled = ordered.repeat((repeats, 1))
        return tiled[:total]

    def _log_sample_grid(self, step: int) -> None:
        if self._visual_labels is None:
            return

        with torch.no_grad():
            labels = self._visual_labels.to(self.device)
            ema_model = self.ema.ema_model
            ema_model.eval()
            samples = ema_model.ddim_sample(
                labels_emb=self.fn_y2h(labels),
                labels=labels,
                shape=(labels.shape[0], self.channels, self.image_size, self.image_size),
                cond_scale=self.cond_scale,
            )

        samples = samples.detach().cpu().clamp(0, 1)
        grid = make_grid(samples, nrow=self._visual_grid_size, normalize=False)

        metrics_payload = self._compute_sampling_metrics(
            labels=self._visual_labels, generated_samples=samples
        )

        logger = self.logger
        if logger is None:
            return

        grid_np = grid.permute(1, 2, 0).cpu().numpy()

        if wandb is not None and isinstance(logger, pl.loggers.WandbLogger):
            if hasattr(logger, "log_image"):
                logger.log_image(
                    key="train/samples",
                    images=[grid_np],
                    caption=[f"step-{step}"],
                )
            log_payload = {"train/sample_grid": wandb.Image(grid_np, caption=f"Step {step}")}
            log_payload.update(metrics_payload)
            logger.experiment.log(log_payload, step=step)
        elif hasattr(logger, "log_image"):
            logger.log_image(
                key="train/samples",
                images=[grid_np],
                caption=[f"step-{step}"],
            )

        if metrics_payload and hasattr(logger, "log_metrics"):
            logger.log_metrics(metrics_payload, step=step)

        if hasattr(logger, "log_metrics"):
            logger.log_metrics({"train/sample_grid_logged": step}, step=step)

    def _compute_sampling_metrics(
        self, *, labels: Tensor, generated_samples: Tensor
    ) -> Dict[str, float]:
        if self.similarity_calculator is None:
            return {}
        # try:
        #     _, metrics = self.similarity_calculator.calculate_all_metrics(labels.detach().cpu(), generated_samples.detach().cpu())
        # except Exception as error:
        #     logging.getLogger(__name__).warning(
        #         "Failed to compute sampling metrics: %s", error
        #     )
        #     return {}

        _, metrics = self.similarity_calculator.calculate_all_metrics(labels.detach().cpu(),
                                                                      generated_samples.detach().cpu())

        return {
            "train/sample_mse_mean": float(metrics["mse"]["mean"]),
            "train/sample_ssim_mean": float(metrics["ssim"]["mean"]),
            "train/sample_psnr_mean": float(metrics["psnr"]["mean"]),
            "train/sample_lpips_mean": float(metrics["lpips"]["mean"]),
        }

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        if self.trainer is None or self.similarity_calculator is None:
            return

        val_loaders = getattr(self.trainer, "val_dataloaders", None)
        if val_loaders is None:
            return

        if isinstance(val_loaders, DataLoader):
            val_loader = val_loaders
        else:
            if len(val_loaders) == 0:
                return
            val_loader = val_loaders[0]

        try:
            batch = next(iter(val_loader))
        except StopIteration:
            return

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            return

        _, cond_inputs = batch
        if not isinstance(cond_inputs, Mapping) or "tensor" not in cond_inputs:
            return

        labels = cond_inputs["tensor"]
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)

        max_examples = min(self.sample_batch_size, labels.shape[0])
        labels = labels[:max_examples]

        if (self.current_epoch + 1) % self.sample_every_n_epochs == 0:
            generated_samples = self.sample_given_labels(labels, batch_size=max_examples)
            generated_samples = generated_samples[:max_examples].clamp(0, 1)

            try:
                _, metrics = self.similarity_calculator.calculate_all_metrics(
                    labels.cpu(), generated_samples.cpu()
                )
            except Exception as error:
                logging.getLogger(__name__).warning(
                    "Failed to compute validation sampling metrics: %s", error
                )
                return

            mse_mean = float(metrics["mse"]["mean"])
            ssim_mean = float(metrics["ssim"]["mean"])
            psnr_mean = float(metrics["psnr"]["mean"])
            lpips_mean = float(metrics["lpips"]["mean"])

            self.log("val/sample_mse_mean", mse_mean, prog_bar=True, on_step=True)
            self.log("val/sample_ssim_mean", ssim_mean, prog_bar=True, on_step=True)
            self.log("val/sample_psnr_mean", psnr_mean, prog_bar=True, on_step=True)
            self.log("val/sample_lpips_mean", lpips_mean, prog_bar=True, on_step=True)

            logger = self.logger
            if logger is not None and hasattr(logger, "log_metrics"):
                logger.log_metrics(
                    {
                        "val/sample_mse_mean": mse_mean,
                        "val/sample_ssim_mean": ssim_mean,
                        "val/sample_psnr_mean": psnr_mean,
                        "val/sample_lpips_mean": lpips_mean,
                    },
                    step=self.global_step,
                )

    # ------------------------------------------------------------------
    # Vicinal sampling helpers
    # ------------------------------------------------------------------
    def _sample_training_batch(
        self, *, batch_size: int, device: torch.device
    ):
        if self._train_images_cpu is None or self._train_labels_cpu is None:
            raise RuntimeError("Training dataset cache has not been initialised")

        dataset_labels = self._train_labels_cpu
        num_examples = dataset_labels.shape[0]

        if self.threshold_type == "hard" and self.kappa == 0 and self.kernel_sigma == 0:
            indices = torch.randint(0, num_examples, (batch_size,))
            batch_images = self._train_images_cpu[indices].to(device)
            batch_labels = dataset_labels[indices].to(device)
            return batch_images, batch_labels, None

        base_indices = torch.randint(0, num_examples, (batch_size,))
        target_labels = dataset_labels[base_indices].clone()
        if self.kernel_sigma > 0:
            target_labels += torch.randn_like(target_labels) * self.kernel_sigma

        distances = self._compute_label_distances(target_labels)

        selected_indices = torch.empty(batch_size, dtype=torch.long)
        vicinal_weights: Optional[Tensor] = (
            torch.ones(batch_size, dtype=distances.dtype) if self.threshold_type == "soft" else None
        )

        for i in range(batch_size):
            dist_row = distances[i]
            if self.threshold_type == "hard":
                if self.kappa <= 0:
                    chosen = torch.argmin(dist_row)
                else:
                    candidates = torch.nonzero(dist_row <= self.kappa, as_tuple=False).view(-1)
                    if candidates.numel() == 0:
                        chosen = torch.argmin(dist_row)
                    else:
                        chosen = candidates[torch.randint(0, candidates.numel(), (1,)).item()]
            else:
                weights = torch.exp(-self.kappa * dist_row**2)
                if self.soft_threshold is not None:
                    weights = torch.where(dist_row <= self.soft_threshold, weights, torch.zeros_like(weights))
                weight_sum = weights.sum()
                if weight_sum <= 0:
                    chosen = torch.argmin(dist_row)
                    weight_value = torch.tensor(1.0, dtype=weights.dtype)
                else:
                    probs = weights / weight_sum
                    chosen = torch.multinomial(probs, 1).item()
                    weight_value = weights[chosen]
                vicinal_weights[i] = weight_value

            selected_indices[i] = chosen

        batch_images = self._train_images_cpu[selected_indices].to(device)
        batch_labels = dataset_labels[selected_indices].to(device)
        if vicinal_weights is not None:
            vicinal_weights = vicinal_weights.to(device)

        return batch_images, batch_labels, vicinal_weights

    def _compute_label_distances(self, targets: Tensor) -> Tensor:
        if self._train_labels_cpu is None:
            raise RuntimeError("Training labels are not cached")

        dataset_labels = self._train_labels_cpu.to(targets)
        targets = targets.to(dataset_labels)

        if self.distance_metric == "euclidean":
            return torch.cdist(targets, dataset_labels)

        assert self.distance_metric == "mahalanobis"
        if self._label_cov_inv is None:
            raise RuntimeError("Mahalanobis metric selected but covariance statistics are unavailable")

        cov_inv = self._label_cov_inv.to(targets)
        diff = targets[:, None, :] - dataset_labels[None, :, :]
        mahal_sq = torch.einsum("bij,jk,bik->bi", diff, cov_inv, diff)
        mahal_sq = torch.clamp(mahal_sq, min=0.0)
        return torch.sqrt(mahal_sq)

    # ------------------------------------------------------------------
    # Public sampling helper
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_given_labels(
        self,
        labels: Tensor,
        *,
        batch_size: Optional[int] = None,
        sampler: str = "ddim",
        cond_scale: Optional[float] = None,
    ) -> Tensor:
        """Generate samples conditioned on the provided labels."""

        if batch_size is None:
            batch_size = self.sample_batch_size

        cond_scale = cond_scale if cond_scale is not None else self.cond_scale

        ema_model = self.ema.ema_model
        ema_model.eval()

        all_images: list[Tensor] = []
        labels = labels.to(self.device)
        labels = self._ensure_label_2d_tensor(labels)
        num_labels = labels.shape[0]
        batch_size = min(batch_size, num_labels)

        for start in range(0, num_labels, batch_size):
            end = start + batch_size
            batch_labels = labels[start:end]
            if sampler == "ddpm":
                images = ema_model.sample(
                    labels_emb=self.fn_y2h(batch_labels),
                    labels=batch_labels,
                    cond_scale=cond_scale,
                )
            else:
                images = ema_model.ddim_sample(
                    labels_emb=self.fn_y2h(batch_labels),
                    labels=batch_labels,
                    shape=(
                        batch_labels.shape[0],
                        self.channels,
                        self.image_size,
                        self.image_size,
                    ),
                    cond_scale=cond_scale,
                )
            all_images.append(images.detach().cpu())

        return torch.cat(all_images, dim=0)

