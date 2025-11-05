import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.vqvae import VQVAE
import wandb


class LightningVQVAE(pl.LightningModule):
    def __init__(self, im_channels, model_config, learning_rate=1e-3, sample_log_interval=5):
        super().__init__()
        self.vqvae = VQVAE(im_channels, model_config)
        self.learning_rate = learning_rate
        self.sample_log_interval = sample_log_interval  # Interval to log samples

    def forward(self, x, context=None):
        return self.vqvae(x.to(self.device), context)

    def encode(self, x, context=None):
        return self.vqvae.encode(x.to(self.device), context)

    def decode(self, x, context=None):
        return self.vqvae.decode(x.to(self.device), context)

    def training_step(self, batch, batch_idx):
        x = batch
        reconstructed, _, quant_losses = self(x)
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # Sum up VQ-VAE losses
        codebook_loss = quant_losses['codebook_loss']
        commitment_loss = quant_losses['commitment_loss']
        total_loss = reconstruction_loss + codebook_loss + commitment_loss

        # Logging the losses
        self.log('train/reconstruction_loss', reconstruction_loss)
        self.log('train/codebook_loss', codebook_loss)
        self.log('train/commitment_loss', commitment_loss)
        self.log('train/total_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        reconstructed, _, quant_losses = self(x)
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # Sum up VQ-VAE losses
        codebook_loss = quant_losses['codebook_loss']
        commitment_loss = quant_losses['commitment_loss']
        total_loss = reconstruction_loss + codebook_loss + commitment_loss

        # Logging validation losses
        self.log('val/reconstruction_loss', reconstruction_loss)
        self.log('val/codebook_loss', codebook_loss)
        self.log('val/commitment_loss', commitment_loss)
        self.log('val/total_loss', total_loss)

    def on_train_epoch_end(self):
        """Generate and log sample reconstructions at the end of each epoch."""
        if self.current_epoch % self.sample_log_interval == 0:
            self.generate_and_plot_samples()

    def generate_and_plot_samples(self):
        """Generate reconstructions for a batch of validation samples."""
        val_batch = next(iter(self.trainer.datamodule.val_dataloader()))  # Get a batch from validation set
        x = val_batch
        self.eval()  # Set model to evaluation mode for inference
        with torch.no_grad():
            reconstructed, z, _ = self(x)

        self._plot_samples(x, z, reconstructed)
        self.train()  # Return model to training mode

    def _plot_samples(self, originals, z, reconstructions, num_samples=3):
        """Helper function to plot original, latent and reconstructed images."""
        originals = originals[:num_samples].cpu()
        z = z[:num_samples].cpu()
        reconstructions = reconstructions[:num_samples].cpu()

        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
        for i in range(num_samples):
            # Plot original images
            axes[0, i].imshow(originals[i].permute(1, 2, 0).squeeze(), cmap="gray")
            axes[0, i].axis("off")
            # Plot reconstructed images
            axes[1, i].imshow(z[i].permute(1, 2, 0).squeeze(), cmap="gray")
            axes[1, i].axis("off")
            # Plot reconstructed images
            axes[2, i].imshow(reconstructions[i].permute(1, 2, 0).squeeze(), cmap="gray")
            axes[2, i].axis("off")

        plt.tight_layout()

        # Log the image to WandB using wandb.Image
        self.logger.experiment.log({f"vqvae_outputs_epoch_{self.current_epoch}": wandb.Image(fig)})

        # Close the figure to free up memory
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
