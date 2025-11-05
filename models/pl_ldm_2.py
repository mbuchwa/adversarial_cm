import os
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import numpy as np
from typing import Optional, Dict, Any
from torchmetrics.image import StructuralSimilarityIndexMeasure  # Import SSIM metric
from utils.evaluation_utils import ImageSimilarityMetrics
from utils.resample import LossAwareSampler, UniformSampler
from torch_ema import ExponentialMovingAverage
from denoiser.karras_denoiser\
    import karras_sample


class LightningLatentDiffusion(pl.LightningModule):
    def __init__(
            self,
            unet_model: nn.Module,
            vae_model: nn.Module,
            diffusion: any,
            schedule_sampler: any,
            learning_rate: float,
            num_timesteps: int,
            plot_example_images_epoch_start: int,
            weight_decay=0.0,
            diffusion_type="karras",  # New parameter to select diffusion type
            sigma_min=0.002,
            sigma_max=80,
            rho=7,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet_model', 'vae_model', 'diffusion'])

        # Models
        self.model = unet_model
        self.vae = vae_model
        self.diffusion = diffusion

        # Training params
        self.plot_example_images_epoch_start = plot_example_images_epoch_start
        self.lr = learning_rate
        self.diffusion_type = diffusion_type
        self.num_timesteps = num_timesteps
        self.weight_decay = weight_decay

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)

        # Loss function and metrics
        self.criterion_mse = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Metrics
        self.similarity_calculator = ImageSimilarityMetrics()

        # Ensure UNet parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        self.similarity_calculator = ImageSimilarityMetrics()

        # ###############
        # # MANUAL OPTIM CODE (to show it performs the same as the automatic including grad clipping
        # self.automatic_optimization = False  # Set manual optimization
        # self.scaler = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training
        # self.optimizer = self.configure_optimizers()
        # self.best_val_loss = float("inf")  # Track the best validation loss
        # self.start_time = datetime.now().strftime("%d_%m_%Y-%H-%M")  # Store the training start time once
        # ################

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.99)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(self, noisy_latents: torch.Tensor, t: torch.Tensor,
                cond_input: Optional[Dict[str, torch.Tensor]] = None):
        return self.model(noisy_latents, t, cond_input)

    def training_step(self, batch, batch_idx, logging=True):
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        t, weights = self.schedule_sampler.sample(images.shape[0], self.device)

        # Compute loss using the NoiseSampler
        losses = self.diffusion.training_losses(self.model, z, t, model_kwargs=cond_input)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        if logging:
            self.log("train/loss", loss, prog_bar=True)

        # # MANUAL OPTIM CODE (to show it performs the same as the automatic including grad clipping
        # # ✅ Zero gradients before backward pass
        # self.optimizer.zero_grad()
        #
        # # ✅ Compute scaled gradients using GradScaler
        # self.scaler.scale(loss).backward()
        #
        # # ✅ Unscale gradients before clipping (MUST DO THIS IN MANUAL OPTIMIZATION!)
        # self.scaler.unscale_(self.optimizer)
        #
        # # ✅ Apply manual gradient clipping (Same as Trainer's `gradient_clip_val=1.0`)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        #
        # # ✅ Perform optimizer step with GradScaler
        # self.scaler.step(self.optimizer)
        # self.scaler.update()  # Updates scaling factor for next step

        self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        images, cond_input = batch
        images = images.float()

        # Store original weights
        self.ema.store(self.model.parameters())

        # Copy EMA weights to the model
        self.ema.copy_to(self.model.parameters())

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        t, weights = self.schedule_sampler.sample(images.shape[0], self.device)

        # Compute loss using the NoiseSampler
        losses = self.diffusion.training_losses(self.model, z, t, model_kwargs=cond_input)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        # Log validation loss
        self.log("val/loss", loss, prog_bar=True, on_step=True)

        # Restore original weights
        self.ema.restore(self.model.parameters())

        return loss

    def test_step(self, batch, batch_idx):
        images, cond_input = batch
        images = images.to(self.device)

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        # Generate samples using EMA weights
        _ = self.generate_samples(z.shape, cond_input, use_ema=True)

    def on_train_epoch_end(self):
        if self.current_epoch < self.plot_example_images_epoch_start:
            return

        val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
        images, cond_input = val_batch
        images = images.float().to(self.device)

        with torch.no_grad():
            im, _ = self.vae.encode(images, None)
            _ = self.generate_samples(shape=im.shape, cond_input=cond_input, use_ema=True)

    def on_validation_end(self) -> None:
        if not self.automatic_optimization:
            """Manually save the best checkpoint when val_loss improves."""
            val_loss = self.trainer.callback_metrics.get("val/loss", None)

            if val_loss is None:
                print("⚠️ val/loss not found in callback metrics!")
                return

            val_loss = val_loss.item()  # Convert from tensor to float

            # Check if the new validation loss is better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss  # Update best loss

                # Use the stored start_time
                checkpoint_path = os.path.join(self.trainer.default_root_dir, "./checkpoints/ldm/",
                                               f"best_model_{self.start_time}.ckpt")
                print(f"✅ New best model found! Saving checkpoint to {checkpoint_path}")
                self.trainer.save_checkpoint(checkpoint_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)  # Ensure compatibility with PyTorch's state_dict
        if hasattr(self.ema, "state_dict") and callable(getattr(self.ema, "state_dict")):
            state["ema"] = self.ema.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        if "ema" in state_dict:
            self.ema.load_state_dict(state_dict["ema"])  # Load EMA weights
            print("✅ EMA weights restored from checkpoint!")
        else:
            print("⚠️ No EMA weights found in checkpoint!")

        # Remove "ema" from state_dict to avoid conflicts with strict loading
        state_dict = {k: v for k, v in state_dict.items() if k != "ema"}

        super().load_state_dict(state_dict, strict=strict, *args, **kwargs)

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
            print("✅ EMA weights successfully loaded!")
        else:
            print("⚠️ No EMA weights found in checkpoint.")

    def get_denoised_latent(self, noisy_latents, t, cond_input=None):
        """
        Wrapper for KarrasDiffusion's get_denoised_latent method.
        """
        return self.diffusion.get_denoised_latent(self.model, noisy_latents, t, cond_input)

    def generate_samples(self, shape, cond_input=None, use_ema=False):
        """
        Generate samples using the NoiseSampler.
        Args:
            shape (tuple): Shape of the latent space (e.g., (channels, height, width)).
            cond_input (dict): Conditioning inputs for the model.
            use_ema (bool): Whether to use EMA weights for sampling.

        Returns:
            torch.Tensor: Decoded samples from the VAE.
        """
        self.eval()  # Ensure the model is in evaluation mode

        if use_ema:
            # Temporarily load EMA weights
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())

        try:
            with torch.no_grad():
                # Generate latent samples via diffusion sampling
                sampled_latents = karras_sample(
                    diffusion=self.diffusion,
                    model=self.model,
                    shape=shape,
                    steps=self.num_timesteps,
                    model_kwargs=cond_input,
                    device=self.device,
                    sigma_min=self.sigma_min,
                    sigma_max=self.sigma_max,
                    rho=self.rho
                )
                generated_images = self.vae.decode(sampled_latents)

        finally:
            if use_ema:
                # Restore original weights
                self.ema.restore(self.model.parameters())

        # Calculate all metrics using the calculate_all_metrics() method
        real_images, metrics = self.similarity_calculator.calculate_all_metrics(cond_input['tensor'], generated_images)

        # Compute the mean across all images for each metric
        mse_mean = metrics['mse']['mean']
        ssim_mean = metrics['ssim']['mean']
        psnr_mean = metrics['psnr']['mean']
        lpips_mean = metrics['lpips']['mean']

        # Log mean metrics to WandB
        wandb.log({
            'ldm_epoch': self.current_epoch,
            'ldm_mse_mean': mse_mean,
            'ldm_ssim_mean': ssim_mean,
            'ldm_psnr_mean': psnr_mean,
            'ldm_lpips_mean': lpips_mean
        })

        # Visualize and log the generated images
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        for i, ax in enumerate(axs):
            ax.imshow(generated_images[i, 0].detach().cpu().numpy(), cmap="gray")
            ax.set_title(f"Cond: {np.around(cond_input['tensor'][i, :].cpu().numpy(), decimals=2)}",
                         fontsize=10)
            ax.axis("off")
        plt.tight_layout()

        # Log the figure
        self.logger.log_image(
            key="LDM_generated_images_epoch",
            images=[wandb.Image(fig)],
        )
        plt.close(fig)

        return generated_images

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
