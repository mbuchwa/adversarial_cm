"""Lightning module reproducing the original CcGAN-AVAR training loop."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

try:  # pragma: no cover - optional dependency
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore

from ema_pytorch import EMA

from models.ccgan_networks import (
    ContinuousConditionalDiscriminator,
    ContinuousConditionalGenerator,
)
from config.geometry import GeometryLabelSchema
from utils.evaluation_utils import ImageSimilarityMetrics


@dataclass
class AVAROptimisationConfig:
    """Optimisation hyper-parameters for the AVAR Lightning module."""

    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    latent_dim: int = 128
    batch_size_disc: int = 16
    batch_size_gene: int = 16
    num_d_steps: int = 1
    num_grad_acc_d: int = 1
    num_grad_acc_g: int = 1
    max_grad_norm: float = 1.0
    betas: Tuple[float, float] = (0.5, 0.999)


class LightningCcGANAVAR(pl.LightningModule):
    """Faithful PyTorch Lightning re-implementation of the public AVAR trainer."""

    def __init__(
        self,
        *,
        train_samples: Tensor,
        train_labels: Tensor,
        eval_labels: Tensor,
        generator: ContinuousConditionalGenerator,
        discriminator: ContinuousConditionalDiscriminator,
        fn_y2h: Callable[[Tensor], Tensor],
        vicinal_params: Dict[str, object],
        aux_loss_params: Dict[str, object],
        optimisation_config: AVAROptimisationConfig,
        sample_shape: Tuple[int, ...],
        loss_type: str = "hinge",
        use_diffaug: bool = False,
        diffaug_policy: str = "color,translation,cutout",
        use_amp: bool = False,
        mixed_precision_type: str = "fp16",
        use_ema: bool = False,
        ema_update_after_step: int = int(1e30),
        ema_update_every: int = 10,
        ema_decay: float = 0.999,
        sample_every_n_steps: int = 0,
        sample_batch_size: int = 64,
        save_images_folder: Optional[str] = None,
        checkpoint_sampling_freq: int = 0,
        geometry_label_schema: Optional[GeometryLabelSchema] = None,
        aux_reg_model: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()

        if train_samples.ndim != 2:
            raise ValueError("train_samples must be provided as a 2D tensor of flattened samples")
        if train_samples.size(0) != train_labels.size(0):
            raise ValueError("train_samples and train_labels must share the same batch dimension")

        labels = train_labels.float().view(train_labels.size(0), -1)
        if labels.min() < 0 or labels.max() > 1:
            raise ValueError("train_labels must be normalised to the [0, 1] interval")

        eval_labels = eval_labels.float().view(eval_labels.size(0), -1)
        if eval_labels.min() < 0 or eval_labels.max() > 1:
            raise ValueError("eval_labels must be normalised to the [0, 1] interval")

        self.label_dim = labels.size(1)
        self.sample_shape = sample_shape
        self.generator = generator
        self.discriminator = discriminator
        self.fn_y2h = fn_y2h
        self.optimisation_config = optimisation_config
        self.loss_type = loss_type
        self.use_diffaug = use_diffaug
        self.diffaug_policy = diffaug_policy
        self.use_amp = use_amp
        self.mixed_precision_type = mixed_precision_type
        self.use_ema = use_ema
        self.sample_every_n_steps = int(sample_every_n_steps)
        self.sample_batch_size = int(sample_batch_size)
        self._sample_grid_cols = max(1, int(math.sqrt(max(1, self.sample_batch_size))))
        self.save_images_folder = save_images_folder
        self.checkpoint_sampling_freq = int(checkpoint_sampling_freq)

        if self.save_images_folder is not None:
            os.makedirs(self.save_images_folder, exist_ok=True)

        self.vicinal_params = vicinal_params
        self.aux_loss_params = aux_loss_params
        self.aux_reg_net = aux_reg_model
        if self.aux_reg_net is not None:
            self.aux_reg_net.eval()

        if geometry_label_schema is None:
            self.geometry_label_schema = GeometryLabelSchema()
        elif isinstance(geometry_label_schema, GeometryLabelSchema):
            self.geometry_label_schema = copy.deepcopy(geometry_label_schema)
        else:
            self.geometry_label_schema = GeometryLabelSchema(**geometry_label_schema)
        self._geometry_one_hot_dim = int(self.geometry_label_schema.one_hot_dim)
        self._geometry_continuous_order = tuple(self.geometry_label_schema.continuous_order)
        self._geometry_continuous_dim = len(self._geometry_continuous_order)
        bounds_sequence = [
            self.geometry_label_schema.continuous_bounds[key]
            for key in self._geometry_continuous_order
        ]
        self._geometry_lower_bounds = tuple(float(bound[0]) for bound in bounds_sequence)
        self._geometry_upper_bounds = tuple(float(bound[1]) for bound in bounds_sequence)
        self._geometry_total_dim = self._geometry_one_hot_dim + self._geometry_continuous_dim
        self._geometry_bounds_slice = slice(
            self._geometry_one_hot_dim,
            self._geometry_one_hot_dim + self._geometry_continuous_dim,
        )

        self.automatic_optimization = False

        self.register_buffer("train_samples_buffer", train_samples.float(), persistent=False)
        self.register_buffer("train_labels_buffer", labels, persistent=False)
        self.register_buffer("eval_labels_buffer", eval_labels, persistent=False)

        unique, inverse, counts = torch.unique(labels, dim=0, return_inverse=True, return_counts=True)
        flat_unique = unique.view(unique.size(0), -1)
        if flat_unique.size(1) == 1:
            order = torch.argsort(flat_unique[:, 0])
        else:
            unique_np = flat_unique.detach().cpu().numpy()
            order_np = np.lexsort(np.flipud(unique_np.T))
            order = torch.from_numpy(order_np).to(unique.device)
        unique = unique.index_select(0, order)
        counts = counts.index_select(0, order)
        self.register_buffer("unique_train_labels", unique, persistent=False)
        self.register_buffer("counts_train_elements", counts.float(), persistent=False)

        cond_probe = self.fn_y2h(labels[:1])
        if cond_probe.ndim != 2:
            raise ValueError("fn_y2h must return a 2D tensor of embeddings")
        cond_probe = cond_probe.view(cond_probe.size(0), -1)
        cond_dim = cond_probe.size(1)
        generator_embedding = getattr(self.generator, "embedding", None)
        if generator_embedding is not None:
            generator_label_dim = getattr(generator_embedding, "label_dim", cond_dim)
        else:
            generator_label_dim = cond_dim

        discriminator_embedding = getattr(self.discriminator, "embedding", None)
        if discriminator_embedding is not None:
            discriminator_label_dim = getattr(discriminator_embedding, "label_dim", cond_dim)
        else:
            discriminator_label_dim = cond_dim
        if cond_dim != generator_label_dim:
            raise ValueError(
                "Generator label embedding dimensionality does not match fn_y2h output"
            )
        if cond_dim != discriminator_label_dim:
            raise ValueError(
                "Discriminator label embedding dimensionality does not match fn_y2h output"
            )

        if self.label_dim == 1:
            self.train_labels_np = self.train_labels_buffer.view(-1).cpu().numpy()
            self.unique_train_labels_np = self.unique_train_labels.view(-1).cpu().numpy()
            self.counts_train_elements_np = self.counts_train_elements.cpu().numpy()
            if self.unique_train_labels_np.size > 1:
                self.min_abs_diff = float(
                    torch.diff(self.unique_train_labels.view(-1)).abs().min().cpu().item()
                )
            else:
                self.min_abs_diff = float("inf")
        else:
            self.train_labels_np = None
            self.unique_train_labels_np = None
            self.counts_train_elements_np = None
            self.min_abs_diff = float("inf")

        self.ema_g: Optional[EMA] = None
        if self.use_ema:
            self.ema_g = EMA(
                self.generator,
                beta=ema_decay,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )

        metrics_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_calculator = ImageSimilarityMetrics(
            device=metrics_device,
            geometry_label_schema=self.geometry_label_schema,
        )

        self._diffaugment = self._load_diffaugment()

    def _load_diffaugment(self):
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "CcGAN-AVAR-main", "DiffAugment_pytorch.py"),
            os.path.join(os.path.dirname(__file__), "..", "CCDM_unified", "DiffAugment_pytorch.py"),
        ]
        for path in candidates:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location("ccgan_diffaugment", path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "DiffAugment"):
                    return module.DiffAugment
        return lambda x, policy="": x

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.optimisation_config.generator_lr,
            betas=self.optimisation_config.betas,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.optimisation_config.discriminator_lr,
            betas=self.optimisation_config.betas,
        )
        return [opt_d, opt_g]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @property
    def device_train_samples(self) -> Tensor:
        return self.train_samples_buffer

    @property
    def device_train_labels(self) -> Tensor:
        return self.train_labels_buffer

    @contextmanager
    def _autocast_context(self):
        if not self.use_amp:
            yield
            return
        if self.device.type == "cuda":
            dtype = torch.float16 if self.mixed_precision_type == "fp16" else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                yield
        elif self.mixed_precision_type == "bf16":
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                yield
        else:
            yield

    def _sample_raw_labels(self, batch_size: int) -> Tensor:
        indices = torch.randint(
            0,
            self.unique_train_labels.size(0),
            (batch_size,),
            device=self.unique_train_labels.device,
        )
        return self.unique_train_labels.index_select(0, indices)

    def _sample_target_labels(self, raw_labels: Tensor) -> Tensor:
        sigma = float(self.vicinal_params.get("kernel_sigma", 0.0))
        raw_labels = raw_labels.view(raw_labels.size(0), -1)
        if raw_labels.size(1) != self.label_dim:
            raise ValueError("raw_labels must match the label dimensionality")

        # Copy the raw labels so that we never mutate the caller's tensor
        sampled = raw_labels.clone()

        # The geometry dataset encodes labels using a configurable schema with
        # a one-hot categorical prefix followed by continuous attributes. When
        # the label dimensionality is large enough to accommodate the schema
        # we treat the blocks separately so that the categorical part stays
        # discrete while the continuous attributes remain within their
        # configured bounds after perturbation.
        if self._geometry_total_dim > 0 and self.label_dim >= self._geometry_total_dim:
            one_hot_dim = self._geometry_one_hot_dim
            cont_dim = self._geometry_continuous_dim
            cont_slice = self._geometry_bounds_slice

            if sigma > 0 and cont_dim > 0:
                cont_noise = torch.randn(sampled.size(0), cont_dim, device=sampled.device) * sigma
                sampled[:, cont_slice] = sampled[:, cont_slice] + cont_noise

            if one_hot_dim > 0:
                shape_indices = torch.argmax(raw_labels[:, :one_hot_dim], dim=1)
                sampled[:, :one_hot_dim] = F.one_hot(
                    shape_indices, num_classes=one_hot_dim
                ).to(sampled.dtype)

            if cont_dim > 0:
                lower_bounds = sampled.new_tensor(self._geometry_lower_bounds)
                upper_bounds = sampled.new_tensor(self._geometry_upper_bounds)
                sampled[:, cont_slice] = torch.max(
                    torch.min(sampled[:, cont_slice], upper_bounds),
                    lower_bounds,
                )

            trailing_start = self._geometry_total_dim
            if self.label_dim > trailing_start:
                sampled[:, trailing_start:] = sampled[:, trailing_start:].clamp(0.0, 1.0)
        else:
            noise = torch.randn_like(sampled) * sigma
            sampled = sampled + noise
            sampled = sampled.clamp(0.0, 1.0)

        return sampled

    def _gather_real_images(self, indices: Tensor) -> Tensor:
        flat = self.device_train_samples.index_select(0, indices)
        return flat.view(indices.size(0), *self.sample_shape)

    def _make_vicinity(
        self,
        targets: Tensor,
        raw_targets: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = targets.size(0)
        device = targets.device

        if self.label_dim > 1 and self.vicinal_params.get("use_ada_vic", False):
            raise NotImplementedError("Adaptive vicinity is only implemented for 1D labels")

        targets = targets.view(batch_size, -1)
        raw_targets = raw_targets.view(batch_size, -1)
        if targets.size(1) != self.label_dim:
            raise ValueError("targets must match the configured label dimensionality")
        if raw_targets.size(1) != self.label_dim:
            raise ValueError("raw_targets must match the configured label dimensionality")

        train_labels = self.device_train_labels
        if train_labels.size(1) != self.label_dim:
            raise RuntimeError("Stored training labels have inconsistent dimensionality")
        threshold_type = self.vicinal_params.get("threshold_type", "hard").lower()

        if not self.vicinal_params.get("use_ada_vic", False):
            diff = train_labels.unsqueeze(0) - targets.unsqueeze(1)

            norm_setting = self.vicinal_params.get("distance_norm", "l2")

            def _pairwise_norm(tensor: Tensor) -> Tensor:
                if isinstance(norm_setting, str):
                    key = norm_setting.lower()
                    if key in {"l2", "euclidean"}:
                        return torch.linalg.norm(tensor, ord=2, dim=-1)
                    if key in {"l1", "manhattan"}:
                        return torch.linalg.norm(tensor, ord=1, dim=-1)
                    if key in {"linf", "infinity", "inf"}:
                        return torch.linalg.norm(tensor, ord=float("inf"), dim=-1)
                    raise ValueError(f"Unsupported distance norm '{norm_setting}'")
                return torch.linalg.norm(tensor, ord=float(norm_setting), dim=-1)

            distances = _pairwise_norm(diff)

            if threshold_type == "hard":
                mask = distances <= float(self.vicinal_params["kappa"])
                radius = float(self.vicinal_params["kappa"])
            else:
                kappa = float(self.vicinal_params["kappa"])
                limit = -math.log(float(self.vicinal_params["nonzero_soft_weight_threshold"])) / kappa
                mask = distances.pow(2) <= limit
                radius = math.sqrt(limit)

            real_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            fake_labels = torch.zeros(batch_size, self.label_dim, device=device)
            for i in range(batch_size):
                candidates = mask[i].nonzero(as_tuple=False).view(-1)
                if candidates.numel() == 0:
                    chosen = distances[i].argmin()
                else:
                    chosen = candidates[torch.randint(0, candidates.numel(), (1,), device=device)]
                real_indices[i] = chosen
                if self.label_dim == 1:
                    lower = (targets[i] - radius).clamp(0.0, 1.0)
                    upper = (targets[i] + radius).clamp(0.0, 1.0)
                    fake_labels[i] = lower + (upper - lower) * torch.rand_like(targets[i])
                else:
                    if radius <= 0:
                        fake_labels[i] = targets[i].clamp(0.0, 1.0)
                    else:
                        direction = torch.randn(self.label_dim, device=device)
                        direction = direction / direction.norm(p=2).clamp_min(1e-8)
                        rand_radius = radius * torch.rand(1, device=device).pow(1.0 / self.label_dim)
                        proposal = targets[i] + direction * rand_radius
                        fake_labels[i] = proposal.clamp(0.0, 1.0)

            real_labels = train_labels.index_select(0, real_indices)

            if threshold_type == "hard":
                real_weights = torch.ones(batch_size, device=device)
                fake_weights = torch.ones(batch_size, device=device)
            else:
                kappa = float(self.vicinal_params["kappa"])
                real_weights = torch.exp(-kappa * _pairwise_norm(real_labels - targets) ** 2)
                fake_weights = torch.exp(-kappa * _pairwise_norm(fake_labels - targets) ** 2)

            kappa_l = torch.full((batch_size,), radius, device=device)
            kappa_r = torch.full((batch_size,), radius, device=device)

            return real_indices, fake_labels, real_labels, real_weights, fake_weights, kappa_l, kappa_r

        # Adaptive vicinity (1D labels only)
        if self.unique_train_labels_np is None or self.counts_train_elements_np is None:
            raise RuntimeError("Adaptive vicinity requires one-dimensional labels")

        batch_real_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        batch_fake_labels = torch.zeros(batch_size, 1, device=device)
        kappa_l_all = torch.zeros(batch_size, device=device)
        kappa_r_all = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            target_y = float(targets[i].item())
            idx_y = int(np.searchsorted(self.unique_train_labels_np, target_y, side="left"))
            kappa_l = kappa_r = float(self.vicinal_params["ada_eps"])
            n_got = 0

            if idx_y <= 0:
                idx_l = idx_r = 0
                n_got = int(self.counts_train_elements_np[idx_r])
                kappa_r = abs(target_y - self.unique_train_labels_np[idx_r]) + float(self.vicinal_params["ada_eps"])
                while n_got < self.vicinal_params["min_n_per_vic"] and idx_r < len(self.unique_train_labels_np) - 1:
                    idx_r += 1
                    n_got += int(self.counts_train_elements_np[idx_r])
                    kappa_r = abs(target_y - self.unique_train_labels_np[idx_r])
            elif idx_y >= len(self.unique_train_labels_np) - 1:
                idx_l = idx_r = len(self.unique_train_labels_np) - 1
                n_got = int(self.counts_train_elements_np[idx_l])
                kappa_l = abs(target_y - self.unique_train_labels_np[idx_l]) + float(self.vicinal_params["ada_eps"])
                while n_got < self.vicinal_params["min_n_per_vic"] and idx_l > 0:
                    idx_l -= 1
                    n_got += int(self.counts_train_elements_np[idx_l])
                    kappa_l = abs(target_y - self.unique_train_labels_np[idx_l])
            else:
                if math.isclose(target_y, self.unique_train_labels_np[idx_y], rel_tol=1e-6, abs_tol=1e-6):
                    idx_l, idx_r = idx_y - 1, idx_y + 1
                    n_got = int(self.counts_train_elements_np[idx_y])
                else:
                    idx_l, idx_r = idx_y - 1, idx_y
                    n_got = 0

                dist_left = abs(target_y - self.unique_train_labels_np[idx_l])
                dist_right = abs(target_y - self.unique_train_labels_np[idx_r])
                while n_got < self.vicinal_params["min_n_per_vic"]:
                    if dist_left <= dist_right:
                        kappa_l = dist_left
                        n_got += int(self.counts_train_elements_np[idx_l])
                        idx_l = max(idx_l - 1, 0)
                        dist_left = abs(target_y - self.unique_train_labels_np[idx_l]) if idx_l >= 0 else float("inf")
                    else:
                        kappa_r = dist_right
                        n_got += int(self.counts_train_elements_np[idx_r])
                        idx_r = min(idx_r + 1, len(self.unique_train_labels_np) - 1)
                        dist_right = abs(target_y - self.unique_train_labels_np[idx_r]) if idx_r <= len(self.unique_train_labels_np) - 1 else float("inf")
                    if dist_left == float("inf") and dist_right == float("inf"):
                        break

            if self.vicinal_params.get("use_symm_vic", False):
                radius = max(kappa_l, kappa_r)
                kappa_l = kappa_r = radius

            cond = (self.train_labels_np >= (target_y - kappa_l)) & (
                self.train_labels_np <= (target_y + kappa_r)
            )
            candidate_np = np.where(cond)[0]
            candidate_indices = torch.from_numpy(candidate_np).to(device)
            if candidate_indices.numel() == 0:
                candidate_indices = torch.arange(len(self.train_labels_np), device=device)
            chosen = candidate_indices[torch.randint(0, candidate_indices.numel(), (1,), device=device)]
            batch_real_indices[i] = chosen

            lb = max(0.0, target_y - kappa_l)
            ub = min(1.0, target_y + kappa_r)
            batch_fake_labels[i, 0] = lb + (ub - lb) * torch.rand(1, device=device)
            kappa_l_all[i] = kappa_l
            kappa_r_all[i] = kappa_r

        real_labels = train_labels.index_select(0, batch_real_indices)
        if threshold_type == "soft" or self.vicinal_params.get("ada_vic_type", "").lower() == "hybrid":
            nu_l = 1.0 / (kappa_l_all ** 2)
            nu_r = 1.0 / (kappa_r_all ** 2)
            target_flat = targets.view(-1)
            real_weights = torch.exp(-nu_l * (real_labels.view(-1) - target_flat) ** 2)
            fake_weights = torch.exp(-nu_r * (batch_fake_labels.view(-1) - target_flat) ** 2)
        else:
            real_weights = torch.ones(batch_size, device=device)
            fake_weights = torch.ones(batch_size, device=device)

        return (
            batch_real_indices,
            batch_fake_labels,
            real_labels,
            real_weights,
            fake_weights,
            kappa_l_all,
            kappa_r_all,
        )

    def _disc_adv_loss(self, real_adv: Tensor, fake_adv: Tensor, real_w: Tensor, fake_w: Tensor) -> Tensor:
        if self.loss_type.lower() == "vanilla":
            real_adv = torch.sigmoid(real_adv)
            fake_adv = torch.sigmoid(fake_adv)
            loss_real = -torch.log(real_adv + 1e-20)
            loss_fake = -torch.log(1 - fake_adv + 1e-20)
        elif self.loss_type.lower() == "hinge":
            loss_real = F.relu(1.0 - real_adv)
            loss_fake = F.relu(1.0 + fake_adv)
        else:  # pragma: no cover - invalid configuration
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return (real_w.view_as(loss_real) * loss_real).mean() + (fake_w.view_as(loss_fake) * loss_fake).mean()

    def _gen_adv_loss(self, adv_out: Tensor) -> Tensor:
        if self.loss_type.lower() == "vanilla":
            adv_out = torch.sigmoid(adv_out)
            return -torch.mean(torch.log(adv_out + 1e-20))
        if self.loss_type.lower() == "hinge":
            return -adv_out.mean()
        raise ValueError(f"Unsupported loss type: {self.loss_type}")  # pragma: no cover

    def _adaptive_huber_loss(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        quantile: float,
        delta: Optional[float],
    ) -> Tensor:
        residuals = (y_true - y_pred).abs()
        if delta is None or delta < 0 or delta > 1:
            delta_val = torch.quantile(residuals.detach(), quantile)
        else:
            delta_val = torch.tensor(delta, device=y_pred.device)
        loss = torch.where(
            residuals < delta_val,
            0.5 * residuals ** 2,
            delta_val * (residuals - 0.5 * delta_val),
        )
        return loss.mean()

    def _disc_aux_reg_loss(
        self,
        real_gt: Tensor,
        real_pred: Tensor,
        fake_gt: Tensor,
        fake_pred: Tensor,
        epsilon: Tensor,
    ) -> Tensor:
        loss_type = self.aux_loss_params["aux_reg_loss_type"].lower()
        if loss_type == "mse":
            reg_loss = (
                (real_pred.view(-1) - real_gt.view(-1)) ** 2
            ).mean() + ((fake_pred.view(-1) - fake_gt.view(-1)) ** 2).mean()
        elif loss_type == "ei_hinge":
            factor = float(self.aux_loss_params["aux_reg_loss_ei_hinge_factor"])
            epsilon = epsilon.view(-1)
            real_abs = (real_pred.view(-1) - real_gt.view(-1)).abs()
            fake_abs = (fake_pred.view(-1) - fake_gt.view(-1)).abs()
            reg_loss = torch.mean(torch.clamp(real_abs - epsilon * factor, min=0.0)) + torch.mean(
                torch.clamp(fake_abs - epsilon * factor, min=0.0)
            )
        elif loss_type == "huber":
            quantile = float(self.aux_loss_params["aux_reg_loss_huber_quantile"])
            delta = self.aux_loss_params.get("aux_reg_loss_huber_delta")
            reg_loss = self._adaptive_huber_loss(real_pred.view(-1), real_gt.view(-1), quantile, delta)
            reg_loss = reg_loss + self._adaptive_huber_loss(fake_pred.view(-1), fake_gt.view(-1), quantile, delta)
        else:  # pragma: no cover - invalid configuration
            raise ValueError("Unsupported auxiliary regression loss type")
        return reg_loss

    def _gen_aux_reg_loss(self, fake_gt: Tensor, fake_pred: Tensor) -> Tensor:
        loss_type = self.aux_loss_params["aux_reg_loss_type"].lower()
        if loss_type in {"mse", "huber"}:
            return ((fake_pred.view(-1) - fake_gt.view(-1)) ** 2).mean()
        if loss_type == "ei_hinge":
            return (fake_pred.view(-1) - fake_gt.view(-1)).abs().mean()
        raise ValueError("Unsupported auxiliary regression loss type")  # pragma: no cover

    def _dre_penalty(self, real_out: Tensor, fake_out: Tensor) -> Tensor:
        softplus_fn = torch.nn.Softplus(beta=1, threshold=20)
        sigmoid_fn = torch.nn.Sigmoid()
        sp_div = torch.mean(sigmoid_fn(fake_out) * fake_out) - torch.mean(softplus_fn(fake_out)) - torch.mean(
            sigmoid_fn(real_out)
        )
        dre_lambda = float(self.aux_loss_params.get("dre_lambda", 0.0))
        penalty = dre_lambda * (torch.mean(fake_out) - 1) ** 2
        return sp_div + penalty

    def _maybe_apply_diffaug(self, images: Tensor) -> Tensor:
        if not self.use_diffaug:
            return images
        return self._diffaugment(images, policy=self.diffaug_policy)

    def _compute_sampling_metrics(
        self,
        labels: Tensor,
        generated_samples: Tensor,
        *,
        prefix: str = "train",
        metric_tag: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Optional[Tensor]]:
        if self.similarity_calculator is None:
            return {}, None

        real_images, metrics = self.similarity_calculator.calculate_all_metrics(
            labels.detach().cpu(), generated_samples.detach().cpu()
        )

        if metric_tag:
            base = f"{prefix}/{metric_tag}_"
        else:
            base = f"{prefix}/"

        payload = {
            f"{base}sample_mse_mean": float(metrics["mse"]["mean"]),
            f"{base}sample_ssim_mean": float(metrics["ssim"]["mean"]),
            f"{base}sample_psnr_mean": float(metrics["psnr"]["mean"]),
            f"{base}sample_lpips_mean": float(metrics["lpips"]["mean"]),
        }

        return payload, real_images

    def _log_samples(
        self,
        labels: Tensor,
        samples: Tensor,
        *,
        tag: str,
        prefix: str = "train",
        metric_tag: Optional[str] = None,
        log_on_step: bool = True,
    ) -> None:
        samples = samples.detach().cpu().clamp(0, 1)
        grid = make_grid(samples, nrow=self._sample_grid_cols, normalize=True)

        metrics_payload, real_images = self._compute_sampling_metrics(
            labels, samples, prefix=prefix, metric_tag=metric_tag
        )

        if self.save_images_folder is not None and (
            self.checkpoint_sampling_freq > 0 and self.global_step % self.checkpoint_sampling_freq == 0
        ):
            save_path = os.path.join(self.save_images_folder, f"{tag}_{self.global_step}.png")
            save_image(samples, save_path, nrow=self._sample_grid_cols, normalize=True)

        real_grid = None
        if real_images is not None:
            real_grid = make_grid(real_images, nrow=self._sample_grid_cols, normalize=True)

        logger = self.logger
        if logger is not None:
            if hasattr(logger, "experiment") and wandb is not None:
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                payload: Dict[str, object] = {
                    f"{prefix}/{tag}_grid": wandb.Image(grid_np, caption=f"Step {self.global_step}")
                }
                if real_grid is not None:
                    real_np = real_grid.permute(1, 2, 0).cpu().numpy()
                    payload[f"{prefix}/{tag}_real_grid"] = wandb.Image(
                        real_np, caption=f"Step {self.global_step}"
                    )
                payload.update(metrics_payload)
                logger.experiment.log(payload, step=self.global_step)

            if hasattr(logger, "log_image"):
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                logger.log_image(
                    key=f"{prefix}/{tag}_grid",
                    images=[grid_np],
                    caption=[f"step-{self.global_step}"],
                )
                if real_grid is not None:
                    real_np = real_grid.permute(1, 2, 0).cpu().numpy()
                    logger.log_image(
                        key=f"{prefix}/{tag}_real_grid",
                        images=[real_np],
                        caption=[f"step-{self.global_step}"],
                    )

            if metrics_payload:
                for key, value in metrics_payload.items():
                    self.log(
                        key,
                        value,
                        on_step=log_on_step,
                        on_epoch=not log_on_step,
                        prog_bar=False,
                    )
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(metrics_payload, step=self.global_step)

    # ------------------------------------------------------------------
    # Training logic
    # ------------------------------------------------------------------
    def sample(self, labels: Tensor, *, use_ema: bool = False) -> Tensor:
        batch_size = labels.size(0)
        latents = torch.randn(batch_size, self.optimisation_config.latent_dim, device=self.device)
        cond = self.fn_y2h(labels.to(self.device))
        generator = self.ema_g.ema_model if use_ema and self.ema_g is not None else self.generator
        samples = generator(latents, cond)
        return samples.view(batch_size, *self.sample_shape)

    def _maybe_log_samples(self) -> None:
        if self.sample_every_n_steps <= 0:
            return
        if self.global_step % self.sample_every_n_steps != 0:
            return

        labels = self._sample_target_labels(
            self._sample_raw_labels(self.sample_batch_size)
        ).to(self.device)
        samples = self.sample(labels, use_ema=False)
        self._log_samples(labels, samples, tag="generator")

        if self.ema_g is not None:
            ema_samples = self.sample(labels, use_ema=True)
            self._log_samples(labels, ema_samples, tag="ema_generator")

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        if self.eval_labels_buffer.numel() == 0:
            return

        max_examples = self.eval_labels_buffer.size(0)
        if self.sample_batch_size > 0:
            max_examples = min(max_examples, self.sample_batch_size)

        labels = self.eval_labels_buffer[:max_examples].to(self.device)

        with torch.no_grad():
            samples = self.sample(labels, use_ema=False)
        self._log_samples(
            labels,
            samples,
            tag="generator",
            prefix="val",
            metric_tag="generator",
            log_on_step=False,
        )

        if self.ema_g is not None:
            with torch.no_grad():
                ema_samples = self.sample(labels, use_ema=True)
            self._log_samples(
                labels,
                ema_samples,
                tag="ema_generator",
                prefix="val",
                metric_tag="ema_generator",
                log_on_step=False,
            )

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        del batch, batch_idx

        opt_d, opt_g = self.optimizers()

        d_loss_val = torch.tensor(0.0, device=self.device)
        d_reg_val = torch.tensor(0.0, device=self.device)
        d_dre_val = torch.tensor(0.0, device=self.device)
        g_loss_val = torch.tensor(0.0, device=self.device)
        g_reg_val = torch.tensor(0.0, device=self.device)
        g_dre_val = torch.tensor(0.0, device=self.device)

        for _ in range(self.optimisation_config.num_d_steps):
            opt_d.zero_grad(set_to_none=True)
            for _ in range(self.optimisation_config.num_grad_acc_d):
                raw_labels = self._sample_raw_labels(self.optimisation_config.batch_size_disc)
                target_labels = self._sample_target_labels(raw_labels)

                (
                    real_indices,
                    fake_labels,
                    real_labels,
                    real_weights,
                    fake_weights,
                    kappa_l,
                    kappa_r,
                ) = self._make_vicinity(target_labels, raw_labels)

                real_images = self._maybe_apply_diffaug(
                    self._gather_real_images(real_indices).to(self.device)
                )

                latents = torch.randn(
                    self.optimisation_config.batch_size_disc,
                    self.optimisation_config.latent_dim,
                    device=self.device,
                )
                fake_images = self.generator(latents, self.fn_y2h(fake_labels.to(self.device)))

                cond_targets = self.fn_y2h(target_labels.to(self.device))

                with self._autocast_context():
                    fake_disc_input = self._maybe_apply_diffaug(fake_images.detach())
                    real_out = self.discriminator(real_images, cond_targets)
                    fake_out = self.discriminator(fake_disc_input, cond_targets)
                    d_loss = self._disc_adv_loss(
                        real_out["adv_output"],
                        fake_out["adv_output"],
                        real_weights.to(self.device),
                        fake_weights.to(self.device),
                    )
                    d_loss_val = d_loss.detach()

                    if self.aux_loss_params.get("weight_d_aux_reg_loss", 0.0) > 0 and self.aux_loss_params.get(
                        "use_aux_reg_branch", False
                    ):
                        if self.aux_loss_params.get("use_aux_reg_model", False) and self.aux_reg_net is not None:
                            fake_gt = self.aux_reg_net(fake_images).detach()
                        else:
                            fake_gt = fake_labels.to(self.device)
                        epsilon = torch.maximum(kappa_l, kappa_r)
                        d_reg = self._disc_aux_reg_loss(
                            target_labels.to(self.device),
                            real_out["reg_output"],
                            fake_gt,
                            fake_out["reg_output"],
                            epsilon,
                        )
                        d_loss = d_loss + self.aux_loss_params["weight_d_aux_reg_loss"] * d_reg
                        d_reg_val = d_reg.detach()

                    if self.aux_loss_params.get("use_dre_reg", False):
                        d_dre = self._dre_penalty(real_out["dre_output"], fake_out["dre_output"])
                        d_loss = d_loss + self.aux_loss_params["weight_d_aux_dre_loss"] * d_dre
                        d_dre_val = d_dre.detach()

                    d_loss = d_loss / float(self.optimisation_config.num_grad_acc_d)

                self.manual_backward(d_loss)

            self.clip_gradients(
                opt_d,
                gradient_clip_val=self.optimisation_config.max_grad_norm,
                gradient_clip_algorithm="norm",
            )
            opt_d.step()

        opt_g.zero_grad(set_to_none=True)
        for _ in range(self.optimisation_config.num_grad_acc_g):
            raw_labels = self._sample_raw_labels(self.optimisation_config.batch_size_gene)
            target_labels = self._sample_target_labels(raw_labels)

            latents = torch.randn(
                self.optimisation_config.batch_size_gene,
                self.optimisation_config.latent_dim,
                device=self.device,
            )
            cond = self.fn_y2h(target_labels.to(self.device))

            with self._autocast_context():
                fake_images = self.generator(latents, cond)
                disc_out = self.discriminator(self._maybe_apply_diffaug(fake_images), cond)
                g_loss = self._gen_adv_loss(disc_out["adv_output"])
                g_loss_val = g_loss.detach()

                if self.aux_loss_params.get("weight_g_aux_reg_loss", 0.0) > 0 and (
                    self.aux_loss_params.get("use_aux_reg_branch", False)
                    or self.aux_loss_params.get("use_aux_reg_model", False)
                ):
                    if self.aux_loss_params.get("use_aux_reg_branch", False):
                        fake_pred = disc_out["reg_output"]
                    elif self.aux_reg_net is not None:
                        fake_pred = self.aux_reg_net(fake_images)
                    else:
                        fake_pred = target_labels.to(self.device)
                    g_reg = self._gen_aux_reg_loss(target_labels.to(self.device), fake_pred)
                    g_loss = g_loss + self.aux_loss_params["weight_g_aux_reg_loss"] * g_reg
                    g_reg_val = g_reg.detach()

                if self.aux_loss_params.get("use_dre_reg", False):
                    g_dre = (disc_out["dre_output"].mean() - 1) ** 2
                    g_loss = g_loss + self.aux_loss_params["weight_g_aux_dre_loss"] * g_dre
                    g_dre_val = g_dre.detach()

                g_loss = g_loss / float(self.optimisation_config.num_grad_acc_g)

            self.manual_backward(g_loss)

        self.clip_gradients(
            opt_g,
            gradient_clip_val=self.optimisation_config.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
        opt_g.step()

        if self.ema_g is not None:
            self.ema_g.update()

        self.log("train/d_loss", d_loss_val, on_step=True, prog_bar=True)
        self.log("train/g_loss", g_loss_val, on_step=True, prog_bar=True)
        self.log("train/d_reg_loss", d_reg_val, on_step=True)
        self.log("train/d_dre_loss", d_dre_val, on_step=True)
        self.log("train/g_reg_loss", g_reg_val, on_step=True)
        self.log("train/g_dre_loss", g_dre_val, on_step=True)

        self._maybe_log_samples()

        return {"loss": g_loss_val.detach()}

