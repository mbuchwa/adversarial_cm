"""Lightning module wrapping the original CcGAN training procedure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchvision.utils import make_grid

try:  # pragma: no cover - optional dependency
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore

from models.ccgan_networks import (
    ContinuousConditionalDiscriminator,
    ContinuousConditionalGenerator,
)
from utils.evaluation_utils import ImageSimilarityMetrics


@dataclass
class VicinalSamplingConfig:
    """Configuration of the vicinal sampling procedure."""

    kernel_sigma: float
    kappa: float
    threshold_type: str = "hard"
    nonzero_soft_weight_threshold: float = 0.0
    circular_labels: bool = True

    def __post_init__(self) -> None:
        if self.threshold_type not in {"hard", "soft"}:
            raise ValueError("threshold_type must be either 'hard' or 'soft'")
        if self.threshold_type == "soft" and self.nonzero_soft_weight_threshold <= 0:
            raise ValueError(
                "nonzero_soft_weight_threshold must be positive when using soft vicinal weights"
            )
        if self.kernel_sigma < 0:
            raise ValueError("kernel_sigma must be non-negative")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")


@dataclass
class OptimisationConfig:
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    batch_size_disc: int = 64
    batch_size_gene: int = 64
    latent_dim: int = 2


class LightningCcGAN(pl.LightningModule):
    """Lightning faithful port of the public CcGAN training loop."""

    def __init__(
        self,
        *,
        train_samples: Tensor,
        train_labels: Tensor,
        generator: ContinuousConditionalGenerator,
        discriminator: ContinuousConditionalDiscriminator,
        vicinal_config: VicinalSamplingConfig,
        optimisation_config: OptimisationConfig,
        sample_shape: Tuple[int, ...],
        sample_every_n_steps: int = 0,
        sample_batch_size: int = 64,
    ) -> None:
        super().__init__()

        if train_samples.ndim != 2:
            raise ValueError("train_samples must be a 2D tensor of flattened samples")

        if train_samples.size(0) != train_labels.size(0):
            raise ValueError("train_samples and train_labels must contain the same number of elements")

        self.save_hyperparameters(
            ignore=["train_samples", "train_labels", "generator", "discriminator"]
        )

        samples = train_samples.float()
        labels = train_labels.float()
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        self.label_dim = labels.size(1)
        self.generator = generator
        self.discriminator = discriminator
        self.vicinal_config = vicinal_config
        self.optimisation_config = optimisation_config
        self.sample_shape = sample_shape
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.sample_batch_size = int(sample_batch_size)
        self._sample_grid_cols = max(1, int(math.sqrt(self.sample_batch_size)))

        self.register_buffer("train_samples_buffer", samples, persistent=False)
        self.register_buffer("train_labels_buffer", labels, persistent=False)

        unique = torch.unique(labels, dim=0)
        if unique.ndim == 1:
            unique = unique.unsqueeze(1)
        self.register_buffer("unique_labels", unique)

        self.automatic_optimization = False

        if self.vicinal_config.threshold_type == "soft":
            self._soft_threshold = (
                -math.log(self.vicinal_config.nonzero_soft_weight_threshold)
                / self.vicinal_config.kappa
            )
        else:
            self._soft_threshold = None

        metrics_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_calculator = ImageSimilarityMetrics(device=metrics_device)

    # ------------------------------------------------------------------
    # Lightning required hooks
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.optimisation_config.generator_lr,
            betas=(0.5, 0.999),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.optimisation_config.discriminator_lr,
            betas=(0.5, 0.999),
        )
        return [opt_d, opt_g], []

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @property
    def device_train_samples(self) -> Tensor:
        return self.train_samples_buffer

    @property
    def device_train_labels(self) -> Tensor:
        return self.train_labels_buffer

    def _wrap_labels(self, labels: Tensor) -> Tensor:
        if self.vicinal_config.circular_labels and self.label_dim == 1:
            return labels.remainder(1.0)
        return labels.clamp(0.0, 1.0)

    def _clip_labels(self, labels: Tensor) -> Tensor:
        return labels.clamp(0.0, 1.0)

    def _sample_raw_labels(self, batch_size: int) -> Tensor:
        indices = torch.randint(
            0,
            self.unique_labels.size(0),
            (batch_size,),
            device=self.unique_labels.device,
        )
        return self.unique_labels.index_select(0, indices)

    def _sample_vicinal_targets(self, raw_labels: Tensor) -> Tensor:
        eps = torch.randn_like(raw_labels) * self.vicinal_config.kernel_sigma
        labels = raw_labels + eps
        return self._wrap_labels(labels)

    def _distance_to_labels(self, targets: Tensor) -> Tensor:
        diff = self.device_train_labels.unsqueeze(0) - targets.unsqueeze(1)
        if self.label_dim == 1:
            if self.vicinal_config.threshold_type == "hard":
                return diff.abs()
            return diff.pow(2)
        if self.vicinal_config.threshold_type == "hard":
            return diff.norm(dim=-1)
        return diff.pow(2).sum(dim=-1)

    def _vicinal_indices(self, target: Tensor) -> Tensor:
        distances = self._distance_to_labels(target.unsqueeze(0)).squeeze(0)
        if self.vicinal_config.threshold_type == "hard":
            mask = distances <= self.vicinal_config.kappa
        else:
            mask = distances <= self._soft_threshold
        return mask.nonzero(as_tuple=False).view(-1)

    def _sample_fake_labels(self, target: Tensor) -> Tensor:
        if self.vicinal_config.threshold_type == "hard":
            radius = self.vicinal_config.kappa
        else:
            radius = math.sqrt(self._soft_threshold)
        lower = self._clip_labels(target - radius)
        upper = self._clip_labels(target + radius)
        noise = torch.rand_like(target)
        return lower + (upper - lower) * noise

    def _gather_real_batch(self, batch_size: int):
        raw_labels = self._sample_raw_labels(batch_size)
        target_labels = self._sample_vicinal_targets(raw_labels)

        real_indices = torch.empty(batch_size, dtype=torch.long, device=self.device)
        fake_labels = torch.empty_like(target_labels)
        for i in range(batch_size):
            candidate = target_labels[i]
            indices = self._vicinal_indices(candidate)
            while indices.numel() == 0:
                candidate = self._sample_vicinal_targets(raw_labels[i : i + 1]).squeeze(0)
                target_labels[i] = candidate
                indices = self._vicinal_indices(candidate)
            choice = indices[torch.randint(0, indices.numel(), (1,), device=indices.device)]
            real_indices[i] = choice.to(self.device)
            fake_labels[i] = self._sample_fake_labels(candidate)

        real_samples = self.device_train_samples.index_select(0, real_indices)
        real_labels = self.device_train_labels.index_select(0, real_indices)
        return real_samples, real_labels, target_labels.to(self.device), fake_labels.to(self.device)

    def _compute_weights(self, real_labels: Tensor, fake_labels: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        if self.vicinal_config.threshold_type == "hard":
            ones = torch.ones(real_labels.size(0), device=real_labels.device)
            return ones, ones
        diff_real = real_labels - targets
        diff_fake = fake_labels - targets
        if self.label_dim == 1:
            sq_real = diff_real.pow(2).view(-1)
            sq_fake = diff_fake.pow(2).view(-1)
        else:
            sq_real = diff_real.pow(2).sum(dim=1)
            sq_fake = diff_fake.pow(2).sum(dim=1)
        weights_real = torch.exp(-self.vicinal_config.kappa * sq_real)
        weights_fake = torch.exp(-self.vicinal_config.kappa * sq_fake)
        return weights_real, weights_fake

    def _discriminator_step(self) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = self.optimisation_config.batch_size_disc
        real_samples, real_labels, targets, fake_labels = self._gather_real_batch(batch_size)

        noise = torch.randn(batch_size, self.optimisation_config.latent_dim, device=self.device)
        fake_samples = self.generator(noise, fake_labels)

        weights_real, weights_fake = self._compute_weights(real_labels, fake_labels, targets)

        real_out = self.discriminator(real_samples, targets)
        fake_out = self.discriminator(fake_samples.detach(), targets)

        d_loss = -torch.mean(weights_real * torch.log(real_out.view(-1) + 1e-20))
        d_loss -= torch.mean(weights_fake * torch.log(1 - fake_out.view(-1) + 1e-20))

        return d_loss, real_out.mean().detach(), fake_out.mean().detach()

    def _generator_step(self) -> Tuple[Tensor, Tensor]:
        batch_size = self.optimisation_config.batch_size_gene
        labels_raw = self._sample_raw_labels(batch_size)
        labels = self._sample_vicinal_targets(labels_raw).to(self.device)
        noise = torch.randn(batch_size, self.optimisation_config.latent_dim, device=self.device)
        fake_samples = self.generator(noise, labels)
        probs = self.discriminator(fake_samples, labels)
        g_loss = -torch.mean(torch.log(probs.view(-1) + 1e-20))
        return g_loss, probs.mean().detach()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):  # type: ignore[override]
        opt_d, opt_g = self.optimizers()

        opt_d.zero_grad(set_to_none=True)
        d_loss, real_prob, fake_prob = self._discriminator_step()
        self.manual_backward(d_loss)
        opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        g_loss, gen_prob = self._generator_step()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("train/d_loss", d_loss, on_step=True, prog_bar=True)
        self.log("train/g_loss", g_loss, on_step=True, prog_bar=True)
        self.log("train/real_prob", real_prob, on_step=True)
        self.log("train/fake_prob", fake_prob, on_step=True)
        self.log("train/gen_prob", gen_prob, on_step=True)

        self._maybe_log_samples()

        return {"loss": g_loss.detach()}

    def sample(self, labels: Tensor, batch_size: Optional[int] = None) -> Tensor:
        if batch_size is None:
            batch_size = labels.size(0)
        noise = torch.randn(batch_size, self.optimisation_config.latent_dim, device=self.device)
        samples = self.generator(noise, labels.to(self.device))
        return samples.view(batch_size, *self.sample_shape)

    def _compute_sampling_metrics(
        self, labels: Tensor, generated_samples: Tensor
    ) -> Tuple[Dict[str, float], Optional[Tensor]]:
        if self.similarity_calculator is None:
            return {}, None

        real_images, metrics = self.similarity_calculator.calculate_all_metrics(
            labels.detach().cpu(), generated_samples.detach().cpu()
        )

        payload = {
            "train/sample_mse_mean": float(metrics["mse"]["mean"]),
            "train/sample_ssim_mean": float(metrics["ssim"]["mean"]),
            "train/sample_psnr_mean": float(metrics["psnr"]["mean"]),
            "train/sample_lpips_mean": float(metrics["lpips"]["mean"]),
        }

        return payload, real_images

    def _maybe_log_samples(self) -> None:
        if self.sample_every_n_steps <= 0:
            return
        if self.global_step % self.sample_every_n_steps != 0:
            return
        if self.logger is None:
            return

        labels = self._sample_vicinal_targets(
            self._sample_raw_labels(self.sample_batch_size)
        ).to(self.device)
        samples = self.sample(labels, self.sample_batch_size).detach().cpu().clamp(0, 1)
        grid = make_grid(samples, nrow=self._sample_grid_cols, normalize=True)

        metrics_payload, real_images = self._compute_sampling_metrics(labels, samples)

        real_grid = None
        if real_images is not None:
            real_grid = make_grid(real_images, nrow=self._sample_grid_cols, normalize=True)

        logger = self.logger
        if hasattr(logger, "experiment") and wandb is not None:
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            log_payload: Dict[str, object] = {
                "train/sample_grid": wandb.Image(grid_np, caption=f"Step {self.global_step}")
            }
            if real_grid is not None:
                real_grid_np = real_grid.permute(1, 2, 0).cpu().numpy()
                log_payload["train/real_grid"] = wandb.Image(
                    real_grid_np, caption=f"Step {self.global_step}"
                )
            log_payload.update(metrics_payload)
            logger.experiment.log(log_payload, step=self.global_step)

        if hasattr(logger, "log_image"):
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            logger.log_image(
                key="train/sample_grid",
                images=[grid_np],
                caption=[f"step-{self.global_step}"],
            )
            if real_grid is not None:
                real_grid_np = real_grid.permute(1, 2, 0).cpu().numpy()
                logger.log_image(
                    key="train/real_grid",
                    images=[real_grid_np],
                    caption=[f"step-{self.global_step}"],
                )

        if metrics_payload:
            for key, value in metrics_payload.items():
                self.log(key, value, prog_bar=False, on_step=True)
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(metrics_payload, step=self.global_step)

