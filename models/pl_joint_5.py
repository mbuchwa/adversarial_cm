import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluation_utils import ImageSimilarityMetrics, ImageEvaluationMetrics
from utils.resample import LossAwareSampler, UniformSampler
from models.pl_consistency_model import LightningConsistencyModel
from models.pl_discriminator import LightningConditionalDiscriminator
from transformers import get_cosine_schedule_with_warmup
import lpips
from utils.rollout_adjustment_helper import DynamicRolloutScheduler
from contextlib import contextmanager
from utils.discriminator_utils import compute_r1_penalty
from typing import List, Optional, Tuple
from torch_ema import ExponentialMovingAverage
import math

@contextmanager
def disable_gradients(module):
    try:
        for p in module.parameters():
            p.requires_grad = False
        yield
    finally:
        for p in module.parameters():
            p.requires_grad = True


class JointDiffusionTrainer(pl.LightningModule):
    def __init__(
            self,
            consistency_model: LightningConsistencyModel,
            discriminator: LightningConditionalDiscriminator,
            vae: nn.Module,
            batch_size: int,
            learning_rate: float = 1e-4,
            learning_rate_disc: float = 1e-4,
            learining_rate_consistency: float = 1e-4,
            discriminator_weight: float = 0.001,
            discriminator_fm_weight: float = 0.00,
            perceptual_weight: float = 0.1,
            discriminator_fm_weigth: float = 0.1,
            adv_ramp_up_epochs: int = 50,
            discriminator_steps: int = 1,
            disc_loss_type: str = "bce",
            disc_input_noise_std: float = 0.0,
            label_smoothing: float = 0.1,
            max_discriminator_steps: int = 5,
            gradient_penalty_weight: float = 1.0,
            use_residual_discriminator: bool = False,
            weight_decay_disc: float = 0.0,
            num_classes: int = 4,
            num_objects: int = 1,
            num_cond_variables: int = 8,
            use_discriminator: bool = True,
            discriminator_start_epoch: int = 0,
            freeze_cm_epoch: int = 0,
            plot_example_images_epoch_start: int = 0,
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
            dataset='geometry',
            max_rollout: int = 3,
            max_rollout_limit: int = 10,
            min_rollout: int = 1,  # CAUTION: values > 1 make CM unable to model last timestep -> Fix
            disc_baseline_reset_epochs: int = 5,
            rollout_schedule: Optional[List[Tuple[int, int]]] = None,
            rollout_plateau_patience: int = 5,
            rollout_plateau_delta: float = 1e-4,
            rollout_plateau_cooldown: int = 1,
            rollout_warmup_epochs: int = 10,
            lambda_adv: float = 0.1,
            lambda_adv_min: float = 0.01,
            lambda_adv_max: float = 2.0,
            lambda_adv_baseline_min: float = 0.05,
    ):
        super().__init__()

        # Models
        self.consistency_model = consistency_model
        self.discriminator = discriminator
        self.vae = vae

        if self.consistency_model.model is not None:
            self.consistency_model.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        # Enable gradients
        for param in self.consistency_model.model.parameters():
            param.requires_grad = True
        for param in self.discriminator.parameters():
            param.requires_grad = True

        self.discriminator_ema = ExponentialMovingAverage(
            self.discriminator.parameters(),
            decay=0.999,
        )

        # Batch size
        self.batch_size = batch_size

        # Training params
        self.learning_rate = learning_rate
        self.learning_rate_disc = learning_rate_disc
        self.learning_rate_consistency = learining_rate_consistency
        self.discriminator_weight = discriminator_weight
        self.discriminator_fm_weight = discriminator_fm_weight
        self.discriminator_steps = discriminator_steps
        self.disc_loss_type = disc_loss_type
        self.disc_input_noise_std = disc_input_noise_std
        if label_smoothing < 0.0:
            raise ValueError("label_smoothing must be >= 0")
        self.max_discriminator_steps = max_discriminator_steps
        self.rollout_schedule = rollout_schedule
        self.use_residual_discriminator = use_residual_discriminator
        self.weight_decay_disc = weight_decay_disc
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.num_cond_variables = num_cond_variables
        self.use_discriminator = use_discriminator
        self.perceptual_weight = perceptual_weight
        self.discriminator_fm_weigth = discriminator_fm_weigth
        self.gradient_penalty_weight = gradient_penalty_weight
        self.discriminator_start_epoch = discriminator_start_epoch
        self.freeze_cm_epoch = freeze_cm_epoch
        self.plot_example_images_epoch_start = plot_example_images_epoch_start

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.dataset = dataset

        # Rollout scheduling
        self.max_rollout = max_rollout
        self.max_rollout_limit = max_rollout_limit
        self.min_rollout = min_rollout
        self.rollout_plateau_patience = rollout_plateau_patience
        self.rollout_plateau_delta = rollout_plateau_delta
        self.rollout_plateau_cooldown = rollout_plateau_cooldown
        self.rollout_warmup_epochs = rollout_warmup_epochs
        self.rollout_plateau_offset = 0
        self._best_disc_epoch_loss: Optional[float] = None
        self._epochs_since_plateau_improvement = 0
        self._last_rollout_plateau_epoch = -1
        self._disc_epoch_loss_sum = 0.0
        self._disc_epoch_loss_count = 0
        self.consistency_model.diffusion.set_rollout_decay(0)

        # Baseline tracking to keep consistency loss low
        self.baseline_consistency_loss = None
        self.disc_baseline_reset_epochs = disc_baseline_reset_epochs

        # placeholder variable for overwritting when disc loss was calculated
        self.consistency_disc_loss = None
        # placeholder variable for overwritting when consistency terms from diffusion loss function
        self.consistency_terms = None

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.disc_loss = nn.BCEWithLogitsLoss(reduction='mean')

        # Label smoothing for discriminator targets
        self.label_smoothing = label_smoothing
        self.pos_label = 1.0 - self.label_smoothing
        self.neg_label = self.label_smoothing

        self.similarity_calculator = ImageSimilarityMetrics()

        self.perceptual_loss = lpips.LPIPS(net='alex').to('cuda:0')  # From lpips library
        self.adv_ramp_up_epochs = adv_ramp_up_epochs

        self.lambda_adv = lambda_adv
        self.lambda_adv_min = lambda_adv_min
        self.lambda_adv_max = lambda_adv_max
        self.lambda_adv_baseline_min = lambda_adv_baseline_min

        self.automatic_optimization = False  # Set manual optimization
        self.scaler_consistency = self.consistency_model.scaler
        self.scaler_discriminator = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training

        optimizers, schedulers = self.configure_optimizers()
        self.consistency_optimizer, self.discriminator_optimizer = optimizers
        # ✅ Extract BOTH schedulers
        self.consistency_scheduler = schedulers[0]["scheduler"]
        self.discriminator_scheduler = schedulers[1]["scheduler"]
        self._schedulers_stored = True

        if use_discriminator:
            # Only one of the discriminator training strategies can be True
            if use_residual_discriminator:
                self.use_denoised_discriminator = False
            else:
                self.use_denoised_discriminator = True

        self.metric_evaluator = ImageEvaluationMetrics(device='cuda:0')
        self.r1_interval = 16

        self.rollout_scheduler = DynamicRolloutScheduler(
            initial_rollout=max_rollout,
            min_rollout=min_rollout,
            max_rollout=max_rollout_limit,
            warmup_epochs=rollout_warmup_epochs,
            # Performance thresholds
            cm_winning_threshold=0.20,  # Gap < 0.12 = CM fooling disc
            avg_fake_upper_limit=0.40,
            avg_fake_lower_limit=0.20,
            disc_too_strong_threshold=0.70,  # Gap > 0.80 = Disc dominating
            # Patience settings for stability
            adjustment_cooldown=8,  # Wait 8 epochs between adjustments
            increase_patience=5,  # Need 5 epochs of CM winning to increase
            decrease_patience=5,  # Need 10 epochs of disc domination to decrease
            history_size=5,  # Average over 8 recent epochs
            min_epochs_at_rollout=5,  # Must stay at each rollout for 15+ epochs
            # Step sizes
            increase_step=1,  # Increase by 1 when triggered
            decrease_step=1,  # Decrease by 1 when triggered
        )

    @property
    def effective_max_rollout(self) -> Optional[int]:
        if self.max_rollout is None:
            return None
        if self.rollout_plateau_offset <= 0:
            return self.max_rollout
        return max(self.max_rollout - self.rollout_plateau_offset, self.min_rollout)

    def update_rollout_from_schedule(self):
        if not self.rollout_schedule:
            return
        for epoch, rollout in self.rollout_schedule:
            if self.current_epoch >= epoch:
                self.max_rollout = rollout
        min_allowed = self.rollout_plateau_offset + self.min_rollout
        if self.max_rollout < min_allowed:
            self.max_rollout = min_allowed
        self.log("train/max_rollout", float(self.max_rollout), prog_bar=True, on_step=False, on_epoch=True)
        if self.effective_max_rollout is not None:
            self.log(
                "train/max_rollout_effective",
                float(self.effective_max_rollout),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):        # Step 1: Train Consistency Model (sets self.consistency_terms)
        if batch_idx == 0:
            self.update_rollout_from_schedule()
            # self._adjust_rollout()

        if self.current_epoch < self.freeze_cm_epoch:
            for param in self.consistency_model.model.parameters():
                param.requires_grad = False
            consistency_loss = self._train_consistency_model(batch, batch_idx, None)
        else:
            for param in self.consistency_model.model.parameters():
                param.requires_grad = True
            consistency_loss = self._train_consistency_model(batch, batch_idx, self.consistency_optimizer)

        # Step 2: Train Discriminator using fresh CM outputs
        if self.use_discriminator and self.current_epoch >= self.discriminator_start_epoch:
            disc_loss_total = 0.0
            for _ in range(self.discriminator_steps):
                disc_loss_total += self._train_discriminator(
                    batch, batch_idx,
                    self.discriminator_optimizer,
                    consistency_terms=self.consistency_terms  # now guaranteed to match batch
                )
            discriminator_loss = disc_loss_total / self.discriminator_steps
            self.log('train/discriminator_loss', discriminator_loss, prog_bar=True)

        # After discriminator training, optionally update rollout more frequently
        # (This makes it more responsive but might be too aggressive)
        if self.use_discriminator and batch_idx % 100 == 0:  # Every 100 batches
            if hasattr(self, 'last_disc_real_mean') and hasattr(self, 'last_disc_fake_mean'):
                new_rollout, info = self.rollout_scheduler.update(
                    current_epoch=self.current_epoch,
                    disc_real_mean=self.last_disc_real_mean.item(),
                    disc_fake_mean=self.last_disc_fake_mean.item(),
                    disc_loss=None,
                )
                self.max_rollout = new_rollout

        return consistency_loss

    def _train_consistency_model(self, batch, batch_idx, optimizer):
        images, cond_input = batch
        images = images.float()

        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        t, weights = self.consistency_model.schedule_sampler.sample(images.shape[0], self.device)

        self.consistency_model.diffusion.set_rollout_decay(self.rollout_plateau_offset)
        effective_max_rollout = self.effective_max_rollout or self.max_rollout

        terms = self.consistency_model.diffusion.consistency_losses(
            self.consistency_model.model,
            z,
            self.consistency_model.num_scales,
            target_model=self.consistency_model.target_model,
            teacher_model=None,  # self.consistency_model.teacher_model,
            teacher_diffusion=None,
            model_kwargs=cond_input,
            rollout=effective_max_rollout,
            unnoised_training_epochs=self.rollout_warmup_epochs,
            current_epoch=self.current_epoch
        )

        self.consistency_terms = terms  # Used later by discriminator
        base_loss = (terms["loss"] * weights).mean()

        # Track baseline before adversarial training
        if self.baseline_consistency_loss is None:
            self.baseline_consistency_loss = base_loss.detach()
        elif self.current_epoch < self.discriminator_start_epoch:
            self.baseline_consistency_loss = 0.9 * self.baseline_consistency_loss + 0.1 * base_loss.detach()
        elif self.current_epoch == self.discriminator_start_epoch + self.disc_baseline_reset_epochs:
            self.baseline_consistency_loss = base_loss.detach()

        loss = base_loss

        if self.use_discriminator and self.current_epoch >= self.discriminator_start_epoch:
            if isinstance(cond_input, dict):
                cond_tensor = cond_input["tensor"]
            elif isinstance(cond_input, torch.Tensor):
                cond_tensor = cond_input
            else:
                raise TypeError(f'cond_input is of wrong type - {type(cond_input)}')

            if self.num_classes > 0:
                class_indices = torch.tensor([
                    range(s, s + self.num_classes)
                    for s in range(0, cond_tensor.shape[1], self.num_cond_variables)
                ]).flatten()
                class_vars = cond_tensor[:, class_indices].long()
            else:
                class_vars = None

            cont_indices = torch.tensor([
                range(s + self.num_classes, s + self.num_cond_variables)
                for s in range(0, cond_tensor.shape[1], self.num_cond_variables)
            ]).flatten()
            cont_vars = cond_tensor[:, cont_indices]

            if self.use_residual_discriminator:
                real_input = terms["x_tn_true_residual"]
                fake_input = terms["x_tn_consistency_residual"]
            elif self.use_denoised_discriminator:
                real_input = terms["x_tn_true"]
                fake_input = terms["x_tn_consistency"]
            else:
                raise ValueError("Unknown discriminator configuration")

            with torch.no_grad():
                _, real_feats = self.discriminator.forward_with_feats(
                    real_input,
                    terms['x_t'],
                    class_vars,
                    cont_vars,
                    terms["t_cur"],
                    terms["t"],
                    fake_input=False
                )

            fake_logits, fake_feats = self.discriminator.forward_with_feats(
                fake_input,
                terms['x_t'],
                class_vars,
                cont_vars,
                terms["t_cur"],
                terms["t"],
                fake_input=True
            )

            fm_loss = 0.0


            for f_fake, f_real in zip(fake_feats, real_feats):
                fm_loss += F.l1_loss(f_fake, f_real)
            fm_loss = fm_loss / max(len(fake_feats), 1)

            adv_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )

            ramp = min(
                1.0,
                max(0, self.current_epoch - self.discriminator_start_epoch)
                / max(1, self.adv_ramp_up_epochs),
            )

            adv_weight = ramp * self.lambda_adv
            fm_weight = self.discriminator_fm_weight

            loss = loss + adv_weight * adv_loss + fm_weight * fm_loss

            if self.current_epoch >= self.freeze_cm_epoch:
                self.log("train/cm_adv_loss", adv_weight * adv_loss, prog_bar=True)
                self.log("train/cm_feature_matching_loss", fm_weight * fm_loss, prog_bar=True)
                self.log("train/cm_adv_ramp", ramp, prog_bar=True)

        perceptual = self.perceptual_loss(terms["x_tn_consistency"], terms["x_tn_true"]).mean()
        loss = loss + self.perceptual_weight * perceptual

        if self.current_epoch >= self.freeze_cm_epoch:
            self.log("train/consistency_loss", base_loss, prog_bar=True)
            self.log("train/cm_perceptual_loss", perceptual, prog_bar=True)
            self.log("train/consistency_total_loss", loss, prog_bar=True)

        ##################################################
        if self.current_epoch >= self.discriminator_start_epoch and self.baseline_consistency_loss is not None:
            if base_loss.detach() > self.baseline_consistency_loss * 1.5:
                floor = max(self.lambda_adv_min, self.lambda_adv_baseline_min)
                self.lambda_adv = max(self.lambda_adv * 0.9, floor)
            else:
                self.lambda_adv = min(self.lambda_adv * 1.01, self.lambda_adv_max)

        if self.current_epoch >= self.freeze_cm_epoch:
            self.log(
                "train/lambda_adv",
                self.lambda_adv,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        ##################################################

        if self.current_epoch >= self.freeze_cm_epoch:
            optimizer.zero_grad()
            self.scaler_consistency.scale(loss).backward()
            self.scaler_consistency.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.consistency_model.parameters(), max_norm=1.0)
            self.scaler_consistency.step(optimizer)
            self.scaler_consistency.update()

            self.consistency_model.ema.update()
            self.consistency_model.update_target_ema()
            self.consistency_scheduler.step()

            self.log("consistency_lr", self.consistency_optimizer.param_groups[0]['lr'], prog_bar=True, on_step=True)

        return loss

    def _train_discriminator(self, batch, batch_idx=0, optimizer=None, validation=False, consistency_terms=None):
        images, _ = batch
        images = images.float()
        cond_input = consistency_terms["model_kwargs"]

        # Class & continuous feature extraction
        class_variables_indices = torch.tensor([
            range(s, s + self.num_classes) for s in range(0, cond_input['tensor'].shape[1], self.num_cond_variables)
        ]).flatten()
        cont_features_indices = torch.tensor([
            range(s + self.num_classes, s + self.num_cond_variables) for s in
            range(0, cond_input['tensor'].shape[1], self.num_cond_variables)
        ]).flatten()

        if len(class_variables_indices) > 0:
            class_vars = cond_input['tensor'][:, class_variables_indices].long()
        else:
            class_vars = None
        cont_features = cond_input['tensor'][:, cont_features_indices]

        # Choose discriminator input
        if self.use_residual_discriminator:
            real_input = consistency_terms["x_tn_true_residual"]
            fake_input = consistency_terms["x_tn_consistency_residual"]
        elif self.use_denoised_discriminator:
            real_input = consistency_terms["x_tn_true"]
            fake_input = consistency_terms["x_tn_consistency"]

        # ✅ CHANGE 1: Reduced input noise for better learning
        if self.disc_input_noise_std > 0 and not validation:
            real_input = real_input + torch.randn_like(real_input) * self.disc_input_noise_std
            fake_input = fake_input + torch.randn_like(fake_input) * self.disc_input_noise_std

        with torch.no_grad():
            real_input_detached = real_input.detach()
            fake_input_detached = fake_input.detach()

        if class_vars is not None:
            class_vars = class_vars.detach()
        cont_features = cont_features.detach()

        t2 = consistency_terms['t_cur'].detach()
        t1 = consistency_terms['t'].detach()
        x_t = consistency_terms['x_t'].detach()

        # Forward pass
        real_pred = self.discriminator(real_input_detached, x_t, class_vars, cont_features, t2, t1, fake_input=False)
        fake_pred = self.discriminator(fake_input_detached, x_t, class_vars, cont_features, t2, t1, fake_input=True)

        real_mean = torch.sigmoid(real_pred).mean()
        fake_mean = torch.sigmoid(fake_pred).mean()
        prob_gap = real_mean - fake_mean

        if validation:
            self.log("val/disc_real_mean", real_mean, prog_bar=True)
            self.log("val/disc_fake_mean", fake_mean, prog_bar=True)
            self.log("val/disc_prob_gap", prob_gap, prog_bar=True)
            self.last_disc_real_mean = real_mean.detach()
            self.last_disc_fake_mean = fake_mean.detach()
        else:
            self.log("train/disc_real_mean", real_mean, prog_bar=True)
            self.log("train/disc_fake_mean", fake_mean, prog_bar=True)
            self.log("train/disc_prob_gap", prob_gap, prog_bar=True)

        # ✅ CHANGE 2: Adaptive loss weighting based on rollout size
        # Larger rollouts = easier discrimination, so reduce weight
        rollout_size = (t2 - t1).float().mean()
        max_rollout = getattr(self, 'max_timestep', 10)
        rollout_weight = 1.0 - (rollout_size / max_rollout) * 0.5  # Scale from 1.0 to 0.5

        if not validation:
            self.log("train/disc_rollout_weight", rollout_weight, prog_bar=False)
            self.log("train/disc_rollout_size", rollout_size, prog_bar=False)

        # Compute discriminator loss
        if self.disc_loss_type == "bce":
            pos_label_tensor = torch.ones_like(real_pred) * self.pos_label
            neg_label_tensor = torch.ones_like(real_pred) * self.neg_label
            d_loss_real = self.disc_loss(real_pred, pos_label_tensor)
            d_loss_fake = self.disc_loss(fake_pred, neg_label_tensor)
            d_loss = (d_loss_real + d_loss_fake) / 2
        elif self.disc_loss_type == "hinge":
            # ✅ CHANGE 3: Hinge loss is more stable for this task
            d_loss_real = F.relu(1.0 - real_pred).mean()
            d_loss_fake = F.relu(1.0 + fake_pred).mean()
            d_loss = (d_loss_real + d_loss_fake) * 0.5
        elif self.disc_loss_type == "wgan":
            d_loss = (fake_pred.mean() - real_pred.mean())
        else:
            raise ValueError(f"Unknown disc_loss_type: {self.disc_loss_type}")

        # ✅ CHANGE 4: Apply adaptive weighting
        d_loss = d_loss * rollout_weight

        # Log individual loss components
        if not validation:
            self.log("train/disc_loss_real", d_loss_real if self.disc_loss_type == "hinge" else d_loss_real.mean(),
                     prog_bar=False)
            self.log("train/disc_loss_fake", d_loss_fake if self.disc_loss_type == "hinge" else d_loss_fake.mean(),
                     prog_bar=False)

        # R1 gradient penalty
        apply_r1 = not validation and (self.global_step % self.r1_interval == 0)
        r1_penalty = torch.tensor(0.0, device=d_loss.device)
        if apply_r1:
            r1_penalty = compute_r1_penalty(
                self.discriminator,
                real_input,
                class_vars,
                cont_features,
                consistency_terms['x_t'],
                consistency_terms['t_cur'],
                consistency_terms['t'],
            )

        if validation:
            self.log("val/disc_r1", r1_penalty, prog_bar=True)
        else:
            self.log("train/disc_r1", r1_penalty, prog_bar=True)

        total_loss = d_loss + self.gradient_penalty_weight * r1_penalty

        if not validation:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # ✅ CHANGE 5: Monitor gradient statistics
            grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.log("train/disc_grad_norm", grad_norm, prog_bar=True)

            # Additional gradient monitoring
            total_grad_l2 = 0.0
            num_params = 0
            for p in self.discriminator.parameters():
                if p.grad is not None:
                    total_grad_l2 += p.grad.norm().item() ** 2
                    num_params += 1
            if num_params > 0:
                avg_grad_norm = (total_grad_l2 / num_params) ** 0.5
                self.log("train/disc_avg_grad_norm", avg_grad_norm, prog_bar=False)

            optimizer.step()

            self.discriminator_ema.update()

            # Step the discriminator scheduler
            if hasattr(self, 'discriminator_scheduler') and self.discriminator_scheduler is not None:
                # For ReduceLROnPlateau, step with the loss metric
                if isinstance(self.discriminator_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.discriminator_scheduler.step(total_loss.detach())
                else:
                    # For other schedulers (warmup + cosine)
                    self.discriminator_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            self.log("train/discriminator_lr", current_lr, prog_bar=True, on_step=True)

            # ✅ CHANGE 6: Log total loss for monitoring
            self.log("train/disc_total_loss", total_loss, prog_bar=False)

        if validation:
            # Track discriminator loss statistics for plateau detection
            self._disc_epoch_loss_sum += float(total_loss.detach().item())
            self._disc_epoch_loss_count += 1

        # Visual inspection
        if batch_idx == 0 and self.current_epoch % 20 == 0:
            with torch.no_grad():
                self.discriminator.plot_images_with_discriminator_output(
                    fake_input, torch.sigmoid(fake_pred), cont_features,
                    timesteps2=consistency_terms['t_cur'], timesteps=consistency_terms['t'],
                    consistency_generated_images=True
                )
                self.discriminator.plot_images_with_discriminator_output(
                    real_input, torch.sigmoid(real_pred), cont_features,
                    timesteps2=consistency_terms['t_cur'], timesteps=consistency_terms['t'],
                    consistency_generated_images=False
                )

        return total_loss

    # def _handle_rollout_plateau(self, avg_loss: float) -> bool:
    #     """Update rollout decay when the discriminator loss plateaus."""
    #     if not self.use_discriminator:
    #         return False
    #
    #     if self.current_epoch < self.rollout_warmup_epochs:
    #         self._best_disc_epoch_loss = avg_loss
    #         self._epochs_since_plateau_improvement = 0
    #         return False
    #
    #     if self._best_disc_epoch_loss is None or (
    #         avg_loss < self._best_disc_epoch_loss - self.rollout_plateau_delta
    #     ):
    #         self._best_disc_epoch_loss = avg_loss
    #         self._epochs_since_plateau_improvement = 0
    #         return False
    #
    #     self._epochs_since_plateau_improvement += 1
    #     if self._epochs_since_plateau_improvement < self.rollout_plateau_patience:
    #         return False
    #
    #     if (self.current_epoch - self._last_rollout_plateau_epoch) < self.rollout_plateau_cooldown:
    #         return False
    #
    #     max_offset = max(self.max_rollout_limit - self.min_rollout, 0)
    #     if self.rollout_plateau_offset >= max_offset:
    #         return False
    #
    #     self.rollout_plateau_offset += 1
    #     self._last_rollout_plateau_epoch = self.current_epoch
    #     self._epochs_since_plateau_improvement = 0
    #     self._best_disc_epoch_loss = avg_loss
    #
    #     min_allowed = self.rollout_plateau_offset + self.min_rollout
    #     if self.max_rollout < min_allowed:
    #         self.max_rollout = min_allowed
    #
    #     self.consistency_model.diffusion.set_rollout_decay(self.rollout_plateau_offset)
    #     return True

    def on_train_epoch_start(self):
        """Store schedulers for manual use"""
        if not hasattr(self, '_schedulers_stored'):
            # Get schedulers from trainer
            if self.trainer and hasattr(self.trainer, 'lr_scheduler_configs'):
                configs = self.trainer.lr_scheduler_configs
                if len(configs) >= 2:
                    self.consistency_scheduler = configs[0].scheduler
                    self.discriminator_scheduler = configs[1].scheduler
                    self._schedulers_stored = True

    def on_validation_epoch_start(self):
        self._disc_epoch_loss_sum = 0.0
        self._disc_epoch_loss_count = 0

    def validation_step(self, batch, batch_idx):
        # Step 1: Validate CM (sets self.consistency_terms)
        consistency_loss = self._validate_consistency_model(batch, batch_idx)
        self.log('val/consistency_loss', consistency_loss, prog_bar=True)

        # Step 2: Validate Discriminator using fresh CM outputs
        if self.use_discriminator and self.current_epoch >= self.discriminator_start_epoch:
            discriminator_loss = self._validate_discriminator(
                batch, batch_idx, consistency_terms=self.consistency_terms
            )
            self.log('val/discriminator_loss', discriminator_loss, prog_bar=True)
            self._adjust_discriminator_training()

        return

    def _validate_consistency_model(self, batch, batch_idx):
        # Apply EMA weights
        self.consistency_model.ema.store()
        self.consistency_model.ema.copy_to()

        images, cond_input = batch
        images = images.float()

        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        t, weights = self.consistency_model.schedule_sampler.sample(images.shape[0], self.device)

        with torch.no_grad():
            self.consistency_model.diffusion.set_rollout_decay(self.rollout_plateau_offset)
            effective_max_rollout = self.effective_max_rollout or self.max_rollout
            terms = self.consistency_model.diffusion.consistency_losses(
                self.consistency_model.model,
                z,
                self.consistency_model.num_scales,
                target_model=self.consistency_model.target_model,
                teacher_model=None,
                teacher_diffusion=None,
                model_kwargs=cond_input,
                rollout=effective_max_rollout,
                unnoised_training_epochs=self.rollout_warmup_epochs,
                current_epoch=self.current_epoch
            )

        self.consistency_terms = terms

        if isinstance(self.consistency_model.schedule_sampler, LossAwareSampler):
            self.consistency_model.schedule_sampler.update_with_local_losses(
                t, terms["loss"].detach()
            )

        # Restore original weights
        self.consistency_model.ema.restore()

        loss = (terms["loss"] * weights).mean()

        if self.use_discriminator and self.current_epoch >= self.discriminator_start_epoch:
            if isinstance(cond_input, dict):
                cond_tensor = cond_input["tensor"]
            elif isinstance(cond_input, torch.Tensor):
                cond_tensor = cond_input
            else:
                raise TypeError(f'cond_input is of wrong type - {type(cond_input)}')

            if self.num_classes > 0:
                class_indices = torch.tensor([
                    range(s, s + self.num_classes)
                    for s in range(0, cond_tensor.shape[1], self.num_cond_variables)
                ]).flatten()
                class_vars = cond_tensor[:, class_indices].long()
            else:
                class_vars = None

            cont_indices = torch.tensor([
                range(s + self.num_classes, s + self.num_cond_variables)
                for s in range(0, cond_tensor.shape[1], self.num_cond_variables)
            ]).flatten()
            cont_vars = cond_tensor[:, cont_indices]

            if self.use_residual_discriminator:
                real_input = terms["x_tn_true_residual"]
                fake_input = terms["x_tn_consistency_residual"]
            elif self.use_denoised_discriminator:
                real_input = terms["x_tn_true"]
                fake_input = terms["x_tn_consistency"]
            else:
                raise ValueError("Unknown discriminator configuration")

            with torch.no_grad():
                _, real_feats = self.discriminator.forward_with_feats(
                    real_input,
                    terms['x_t'],
                    class_vars,
                    cont_vars,
                    terms["t_cur"],
                    terms["t"],
                    fake_input=False
                )

            fake_logits, fake_feats = self.discriminator.forward_with_feats(
                fake_input,
                terms['x_t'],
                class_vars,
                cont_vars,
                terms["t_cur"],
                terms["t"],
                fake_input=True
            )

            fm_loss = 0.0


            for f_fake, f_real in zip(fake_feats, real_feats):
                fm_loss += F.l1_loss(f_fake, f_real)
            fm_loss = fm_loss / max(len(fake_feats), 1)

            adv_loss = F.binary_cross_entropy_with_logits(
                fake_logits, torch.ones_like(fake_logits)
            )

            ramp = min(
                1.0,
                max(0, self.current_epoch - self.discriminator_start_epoch)
                / max(1, self.adv_ramp_up_epochs),
            )

            adv_weight = ramp * self.lambda_adv
            fm_weight = self.discriminator_fm_weight

            loss = loss + adv_weight * adv_loss + fm_weight * fm_loss

            if self.current_epoch >= self.freeze_cm_epoch:
                self.log("val/cm_adv_loss", adv_weight * adv_loss, prog_bar=True)
                self.log("val/cm_feature_matching_loss", fm_weight * fm_loss, prog_bar=True)
                self.log("val/cm_adv_ramp", ramp, prog_bar=True)

        perceptual = self.perceptual_loss(terms["x_tn_consistency"], terms["x_tn_true"]).mean()
        loss = loss + self.perceptual_weight * perceptual

        if self.current_epoch >= self.freeze_cm_epoch:
            self.log("val/consistency_loss", loss, prog_bar=True)
            self.log("val/cm_perceptual_loss", perceptual, prog_bar=True)
            self.log("val/consistency_total_loss", loss, prog_bar=True)


        return loss

    def _validate_discriminator(self, batch, batch_idx, consistency_terms):
        return self._train_discriminator(batch, batch_idx, optimizer=None, validation=True,
                                         consistency_terms=consistency_terms)  # Reuse training logic without gradients

    def on_validation_epoch_end(self):
        """log train dis"""
        avg_loss = None
        if self._disc_epoch_loss_count > 0:
            avg_loss = self._disc_epoch_loss_sum / self._disc_epoch_loss_count
            self.log("val/disc_epoch_loss", float(avg_loss), prog_bar=False)

        # Get discriminator performance metrics
        if hasattr(self, 'last_disc_real_mean') and hasattr(self, 'last_disc_fake_mean'):
            # Update rollout using dynamic scheduler
            new_rollout, info = self.rollout_scheduler.update(
                current_epoch=self.current_epoch,
                disc_real_mean=self.last_disc_real_mean.item(),
                disc_fake_mean=self.last_disc_fake_mean.item(),
                disc_loss=avg_loss if avg_loss is not None else None,
            )

            # Update max_rollout
            old_rollout = self.max_rollout
            self.max_rollout = new_rollout

            # Log comprehensive metrics
            self.log("val/rollout_current", float(new_rollout), prog_bar=True)
            self.log("val/rollout_gap", info['gap'], prog_bar=True)

            if 'avg_gap' in info:
                self.log("val/rollout_avg_gap", info['avg_gap'], prog_bar=True)

            # Log consecutive epoch counters
            self.log("val/epochs_cm_winning", float(info.get('epochs_cm_winning', 0)), prog_bar=False)
            self.log("val/epochs_disc_strong", float(info.get('epochs_disc_too_strong', 0)), prog_bar=False)
            self.log("val/epochs_at_rollout", float(info.get('epochs_at_rollout', 0)), prog_bar=True)

            if info.get('changed', False):
                direction = "📈 INCREASED" if new_rollout > old_rollout else "📉 DECREASED"
                print(f"\n{'=' * 60}")
                print(f"{direction} Rollout: {old_rollout} -> {new_rollout}")
                print(f"  Reason: {info['action']}")
                print(f"  Avg Gap: {info.get('avg_gap', 'N/A'):.4f}")
                print(f"  Real: {self.last_disc_real_mean.item():.4f}, Fake: {self.last_disc_fake_mean.item():.4f}")
                if 'epochs_cm_winning' in info:
                    print(f"  Epochs CM Winning: {info['epochs_cm_winning']}")
                if 'epochs_disc_too_strong' in info:
                    print(f"  Epochs Disc Too Strong: {info['epochs_disc_too_strong']}")
                print(f"{'=' * 60}\n")

                self.log("val/rollout_changed", 1.0, prog_bar=True)

            # Log action as numeric for tracking
            action_map = {
                'warmup': 0,
                'cooldown': 1,
                'increase_cm_winning': 2,
                'decrease_disc_too_strong': 3,
                'maintain': 4,
                'insufficient_history': 5,
                'min_epochs_not_met': 6,
            }
            self.log("val/rollout_action", float(action_map.get(info['action'], 10)), prog_bar=False)

        # Update diffusion with current rollout (no decay offset needed anymore)
        self.consistency_model.diffusion.set_rollout_decay(0)

        # Calculate effective rollout for logging
        effective_rollout = self.max_rollout
        self.log("val/max_rollout", float(self.max_rollout), prog_bar=True)
        self.log("val/max_rollout_effective", float(effective_rollout), prog_bar=True)

        # Log scheduler statistics periodically
        if self.current_epoch % 10 == 0:
            stats = self.rollout_scheduler.get_statistics()
            print(f"\n📊 Rollout Statistics (Epoch {self.current_epoch}):")
            print(f"  Avg Gap: {stats['avg_gap']}")
            print(f"  Avg Fake: {stats['avg_fake']}")
            print(f"  Total Adjustments: {stats['total_adjustments']}")
            print(f"  Increases: {stats['increases']}")
            print(f"  Decreases: {stats['decreases']}")
            print(f"  Current Rollout: {stats['current_rollout']}")

            self.log("val/rollout_total_adjustments", float(stats['total_adjustments']), prog_bar=False)
            self.log("val/rollout_increases", float(stats['increases']), prog_bar=False)
            self.log("val/rollout_decreases", float(stats['decreases']), prog_bar=False)

        # # Save discriminator after 200 epochs
        # if self.current_epoch == 200:
        #     import os
        #     checkpoint_dir = "checkpoints/disc"
        #     os.makedirs(checkpoint_dir, exist_ok=True)
        #
        #     checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
        #
        #     # Save only discriminator state
        #     torch.save({
        #         'discriminator_state_dict': self.discriminator.state_dict(),
        #         'discriminator_ema': self.discriminator_ema.state_dict() if hasattr(self, 'discriminator_ema') else None,
        #     }, checkpoint_path)
        #
        #     print(f"✅ Discriminator checkpoint saved to {checkpoint_path}")
        #     self.log("train/discriminator_saved", 1.0, prog_bar=True)

        """Consistency model generation"""
        # Update target model with EMA after validation
        self.consistency_model.update_target_ema()

        if self.current_epoch <= self.plot_example_images_epoch_start and self.current_epoch % 5 != 0:
            return

        val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
        images, cond_input = val_batch
        images = images.float().to(self.device)

        if self.dataset != 'geometry':
            generate_true_images = False
        else:
            generate_true_images = True

        with torch.no_grad():  # , torch.cuda.amp.autocast(enabled=False):
            im, _ = self.vae.encode(images, None)
            generated_images, _ = self.consistency_model.generate_samples(shape=im.shape, cond_input=cond_input,
                                                                          generate_true_images=generate_true_images,
                                                                          return_latent=True)

        # Ensure images are in [0, 1] range
        images = images.clamp(0, 1)
        generated_images = generated_images.clamp(0, 1)

        metrics = self.metric_evaluator.compute_metrics(generated_images, images)
        self.log_dict({
            "val/SFID": metrics.get("sfid", -1),
            # "val/NIQE": metrics.get("niqe", -1),
            "val/Diversity": metrics.get("diversity", -1),
        }, prog_bar=True)

    # def _adjust_rollout(self):
    #     if not hasattr(self, "last_disc_real_mean"):
    #         return
    #     if self._epochs_since_plateau_improvement < self.rollout_plateau_patience:
    #         return
    #     gap = abs(self.last_disc_real_mean.item() - self.last_disc_fake_mean.item())
    #     if gap < 0.1:
    #         self.max_rollout = min(self.max_rollout + 1, self.max_rollout_limit)
    #     elif gap > 0.85:
    #         self.max_rollout = max(self.max_rollout - 1, self.min_rollout)
    #     min_allowed = self.rollout_plateau_offset + self.min_rollout
    #     if self.max_rollout < min_allowed:
    #         self.max_rollout = min_allowed
    #
    #     self.log("train/max_rollout", float(self.max_rollout), prog_bar=True, on_step=True)
    #     if self.effective_max_rollout is not None:
    #         self.log(
    #             "train/max_rollout_effective",
    #             float(self.effective_max_rollout),
    #             prog_bar=True,
    #         )

    def _adjust_discriminator_training(self):
        """Dynamically adjust discriminator steps and adversarial weight."""
        if not hasattr(self, "last_disc_real_mean"):
            return

        real_score = self.last_disc_real_mean.item()
        fake_score = self.last_disc_fake_mean.item()
        gap = abs(real_score - fake_score)

        if fake_score < 0.05 and gap > 0.8:
            self.discriminator_steps = max(self.discriminator_steps - 1, 1)
            self.lambda_adv *= 1.1
        elif gap < 0.1 or (real_score < 0.6 and fake_score > 0.4):
            self.discriminator_steps = min(self.discriminator_steps + 1, self.max_discriminator_steps)
            self.lambda_adv *= 0.95
        elif 0.2 < gap < 0.6 and 0.6 < real_score < 0.9:
            pass

        self.lambda_adv = float(
            max(
                min(self.lambda_adv, self.lambda_adv_max),
                self.lambda_adv_min,
            )
        )

        self.log("train/discriminator_steps", float(self.discriminator_steps), prog_bar=True, on_step=True)
        self.log("train/lambda_adv", self.lambda_adv, prog_bar=True, on_step=True)

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.consistency_model, "ema"):
            checkpoint["ema"] = self.consistency_model.ema.state_dict()
        if hasattr(self, "discriminator_ema"):
            checkpoint["discriminator_ema"] = self.discriminator_ema.state_dict()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)

        # Store consistency model EMA
        if hasattr(self.consistency_model, "ema") and hasattr(self.consistency_model.ema, "state_dict"):
            state["ema"] = self.consistency_model.ema.state_dict()

        # Store discriminator EMA
        if hasattr(self, "discriminator_ema"):
            state["discriminator_ema"] = self.discriminator_ema.state_dict()

        return state

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        # Load consistency model EMA
        if "ema" in state_dict and hasattr(self.consistency_model, "ema"):
            self.consistency_model.ema.load_state_dict(state_dict["ema"])
            print("✅ Consistency model EMA weights restored from checkpoint!")
        else:
            print("⚠️ No consistency model EMA weights found in checkpoint!")

        # Load discriminator EMA
        if "discriminator_ema" in state_dict and hasattr(self, "discriminator_ema"):
            self.discriminator_ema.load_state_dict(state_dict["discriminator_ema"])
            print("✅ Discriminator EMA weights restored from checkpoint!")

        # Remove custom keys before loading into parent
        state_dict = {k: v for k, v in state_dict.items() if k not in {"ema", "discriminator_ema"}}

        super().load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint and hasattr(self.consistency_model, "ema"):
            self.consistency_model.ema.load_state_dict(checkpoint["ema"])
            print("✅ Consistency model EMA weights successfully loaded!")
        else:
            print("⚠️ No consistency model EMA weights found in checkpoint.")

        if "discriminator_ema" in checkpoint and hasattr(self, "discriminator_ema"):
            self.discriminator_ema.load_state_dict(checkpoint["discriminator_ema"])
            print("✅ Discriminator EMA weights successfully loaded!")

    def configure_optimizers(self):
        # Create optimizers
        consistency_optimizer = torch.optim.AdamW(
            self.consistency_model.model.parameters(),
            lr=self.learning_rate_consistency,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.learning_rate_disc,
            betas=(0.5, 0.999),  # (0.0, 0.99),
            weight_decay=self.weight_decay_disc,
            eps=1e-8
        )

        # Default steps if not set
        total_steps = getattr(self, 'total_steps', 250000 / self.batch_size)
        num_warmup_steps = getattr(self, 'num_warmup_steps', 25000 / self.batch_size)

        # Consistency model: Cosine schedule with warmup
        consistency_scheduler = get_cosine_schedule_with_warmup(
            consistency_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )

        # ✅ Discriminator: Use linear warmup + cosine annealing
        # This is more stable than ReduceLROnPlateau for adversarial training
        disc_warmup_steps = int(num_warmup_steps * 0.01)  # Shorter warmup for disc
        disc_total_steps = int(total_steps)  # Shorter warmup for disc

        def disc_lr_lambda(step):
            if step < disc_warmup_steps:
                # Linear warmup
                disc_lambda = float(step) / float(max(1, disc_warmup_steps))
            else:
                # Cosine decay
                progress = float(step - disc_warmup_steps) / float(max(1, disc_total_steps - disc_warmup_steps))
                disc_lambda = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return disc_lambda

        discriminator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            discriminator_optimizer,
            lr_lambda=disc_lr_lambda
        )

        optimizers = [consistency_optimizer, discriminator_optimizer]
        schedulers = [
            {
                "scheduler": consistency_scheduler,
                "interval": "step",
                "frequency": 1
            },
            {
                "scheduler": discriminator_scheduler,
                "interval": "step",  # ✅ Step-based, not epoch-based
                "frequency": 1
            }
        ]

        return optimizers, schedulers

    @staticmethod
    def check_gradients(model):
        """Check gradients for all parameters in the model."""
        print("\nChecking Model parameters:")
        total_params = 0
        params_with_grad = 0

        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad:
                params_with_grad += 1
            print(f"Layer: {name}")
            print(f"  requires_grad: {param.requires_grad}")
            print(f"  grad: {param.grad is not None}")
            if param.grad is not None:
                print(f"  grad norm: {param.grad.norm().item()}")
                print(f"  grad mean: {param.grad.mean().item()}")

        print(f"\nSummary:")
        print(f"Total parameters: {total_params}")
        print(f"Parameters with requires_grad=True: {params_with_grad}")
        print(f"Percentage trainable: {(params_with_grad / total_params) * 100:.2f}%")

    @staticmethod
    def log_model_weights(model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} - Weight mean: {param.data.mean().item()}, Weight norm: {param.data.norm().item()}")




