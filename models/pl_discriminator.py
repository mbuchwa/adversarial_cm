import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.discriminator import ConditionalDiscriminator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torchvision.transforms as T
from typing import Optional, Callable


class LightningConditionalDiscriminator(pl.LightningModule):
    def __init__(
            self,
            vae_model: nn.Module,
            scheduler: any,
            image_channels: int,
            num_classes: int,
            num_continuous_features: int,
            use_noisy_images=False,
            embedding_dim=64,
            num_blocks=4,
            initial_features=64,
            num_objects=1,
            num_scales=10,
            lr=1e-4,
            plot_images_start_epoch=0,
            log_images_with_results_epoch=70,
            discriminator_loss_upper_bound=0.5,
            disc_input_noise_std: float = 0.0,
            smoothing_max_kernel=7,
            smoothing_min_kernel=1,
            smoothing_decay=0.98,
            initial_sigma=2.0,
            min_sigma=0.1,
            gradient_penalty_weight: float = 10.0,
            pos_label: float = 0.9,
            neg_label: float = 0.1,
            generator: Optional[Callable] = None,
            optimizer_betas=(0.9, 0.999),
            weight_decay: float = 0.0,
            use_lr_scheduler: bool = True,
            lr_scheduler_config: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['generator'])
        self.num_classes = num_classes
        self.log_images_with_results_epoch = log_images_with_results_epoch
        self.use_noisy_images = use_noisy_images

        self.plot_images_start_epoch = plot_images_start_epoch

        self.smoothing_max_kernel = smoothing_max_kernel  # Max blur kernel at start
        self.smoothing_min_kernel = smoothing_min_kernel  # Min blur kernel later
        self.smoothing_decay = smoothing_decay  # Decay factor per epoch
        self.disc_input_noise_std = disc_input_noise_std

        self.initial_sigma = initial_sigma  # Strongest blur at start
        self.min_sigma = min_sigma  # Minimum blur intensity

        self.vae = vae_model
        self.scheduler = scheduler

        self.model = ConditionalDiscriminator(
            image_channels=image_channels,
            num_classes=num_classes,
            num_continuous_features=num_continuous_features,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            initial_features=initial_features,
            num_objects=num_objects,
            max_timestep=num_scales,
            timestep_prediction=True,
            double_timestep_prediction=True
        )

        # Loss and regularization
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.discriminator_loss_upper_bound = discriminator_loss_upper_bound
        self.disc_input_noise_std = disc_input_noise_std
        self.gradient_penalty_weight = gradient_penalty_weight
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.generator = generator

        # Optimizer and scheduler configuration
        self.optimizer_betas = optimizer_betas
        self.weight_decay = weight_decay
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config or {}

    def forward(self, x_tn, x_t, class_vector, continuous_features, timestep_1=None, timestep_2=None, fake_input=False):
        return self.model(x_tn, x_t, class_vector, continuous_features, timestep_1, timestep_2, fake_input=fake_input)

    def forward_with_feats(self, x_tn, x_t, class_vector, continuous_features, timestep_1=None, timestep_2=None,
                           fake_input=False):
        return self.model.forward_with_feats(x_tn, x_t, class_vector, continuous_features, timestep_1, timestep_2,
                                             fake_input=fake_input)

    def plot_images_with_discriminator_output(self, images, disc_output, cond,
                                              timesteps2=None, timesteps=None, consistency_generated_images=False):
        """Log a 1x2 grid of images with the discriminator output and condition vector as title using WandB."""
        fig, axes = plt.subplots(1, 2, figsize=(9, 7))

        for i, ax in enumerate(axes.flat):
            if i >= images.size(0):
                break

            # Detach, move to CPU, and convert to NumPy for both outputs and conditions
            img = images[i].permute(1, 2, 0).cpu().numpy()
            discriminator_output = disc_output[i].detach().cpu().item()
            condition_vector = np.round(cond[i].detach().cpu().numpy(), decimals=2)

            # Plot the image and set the title with the discriminator output and condition vector
            ax.imshow(img, cmap='gray')

            if consistency_generated_images:
                if timesteps2 is not None and timesteps is not None:
                    ax.set_title(f'Consistency Generated Image - Real: {discriminator_output:.4f}\n Cond: {condition_vector} \n'
                                 f'Timesteps: {[np.round(timesteps2[i].item(), 4), np.round(timesteps[i].item(), 4)]}')
                else:
                    ax.set_title(
                        f'Consistency Generated Image - Real: {discriminator_output:.4f}\n Cond: {condition_vector}')
                ax.axis('off')
                key = "disc_on_consistency_generated_images"
            else:
                # Plot the image and set the title with the discriminator output and condition vector
                if timesteps2 is not None and timesteps is not None:
                    ax.set_title(f'True Image - Real: {discriminator_output:.4f}\n Cond: {condition_vector} \n'
                                 f'Timesteps: {[np.round(timesteps2[i].item(), 4), np.round(timesteps[i].item(), 4)]}')
                else:
                    ax.set_title(f'True Image - Real: {discriminator_output:.4f}\n Cond: {condition_vector}')
                ax.axis('off')
                key = "disc_on_images"

        plt.tight_layout()

        # Log the image to WandB
        self.logger.log_image(
            key=key,
            images=[wandb.Image(fig)],
        )

        # Close the figure to free up memory
        plt.close(fig)

    def get_smoothing_kernel(self):
        """Gradually decrease kernel size over epochs (ensures an odd value)."""
        kernel_size = int(self.smoothing_max_kernel * (self.smoothing_decay ** self.current_epoch))
        kernel_size = max(self.smoothing_min_kernel, kernel_size)  # Prevent too-small kernel
        return kernel_size if kernel_size % 2 else kernel_size + 1  # Keep it odd

    def get_sigma(self):
        """Decrease sigma value over time for a smoother transition."""
        sigma = max(self.min_sigma, self.initial_sigma * (self.smoothing_decay ** self.current_epoch))
        return sigma

    # def apply_smoothing(self, images):
    #     """Applies Gaussian blur with dynamic kernel size and sigma."""
    #     kernel_size = self.get_smoothing_kernel()
    #     sigma = self.get_sigma()
    #
    #     # Apply blur with decreasing strength over time
    #     smoothing = T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    #     return smoothing(images)

    def training_step(self, batch, batch_idx):
        images, cond_input = batch

        if self.use_noisy_images:
            # Sample random timesteps
            t = torch.randint(0, self.scheduler.num_timesteps, (images.shape[0],)).to(self.device)
            noise = torch.randn_like(images)
            im = self.scheduler.add_noise(images, noise, t)
        else:
            im = images

        if self.disc_input_noise_std > 0:
            im = im + torch.randn_like(im) * self.disc_input_noise_std

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

        # Prepare real labels and fake labels with label smoothing
        real_labels = torch.full((im.size(0), 1), self.pos_label, device=self.device)
        fake_labels = torch.full((im.size(0), 1), self.neg_label, device=self.device)

        # Real condition (unchanged for real noise)
        real_condition = cond_input['tensor']

        # Generate fake images using provided generator (consistency model or other)
        with torch.no_grad():
            if self.generator is not None:
                fake_im = self.generator(real_condition)
                if fake_im.shape != im.shape:
                    _, fake_im, _ = self.vae(fake_im, None)
            else:
                fake_im = torch.randn_like(im)

        # Step 3: Pass both real and generated fake images to the discriminator
        prob_real = self.forward(
            im,
            im,
            real_condition[:, :self.num_classes].long(),
            real_condition[:, self.num_classes:]
        )

        prob_fake = self.forward(
            fake_im,
            fake_im,
            real_condition[:, :self.num_classes].long(),
            real_condition[:, self.num_classes:]
        )

        # Step 4: Compute discriminator loss
        d_loss_real = self.criterion(prob_real, real_labels)
        d_loss_fake = self.criterion(prob_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2

        if self.gradient_penalty_weight > 0:
            im_gp = im.detach().requires_grad_(True)
            real_pred_gp = self.forward(
                im_gp,
                im_gp,
                real_condition[:, :self.num_classes].long(),
                real_condition[:, self.num_classes:]
            )
            gradients = torch.autograd.grad(
                outputs=real_pred_gp.sum(),
                inputs=im_gp,
                create_graph=True,
            )[0]
            r1 = gradients.pow(2).view(gradients.size(0), -1).sum(1).mean()
            d_loss = d_loss + self.gradient_penalty_weight * r1

        # Log the loss and discriminator outputs
        self.log('train/d_loss', d_loss, on_step=True, prog_bar=True)
        self.log('train/disc_real_mean', torch.sigmoid(prob_real).mean(), prog_bar=True)
        self.log('train/disc_fake_mean', torch.sigmoid(prob_fake).mean(), prog_bar=True)

        if self.trainer is not None and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log(
                'train/disc_lr',
                torch.tensor(current_lr, device=im.device),
                on_step=True,
                prog_bar=False,
            )

        # Plot images after epoch 50 for the first batch
        if self.current_epoch > self.plot_images_start_epoch and batch_idx == 0:
            with torch.no_grad():
                discriminator_output_real = torch.sigmoid(prob_real).squeeze(1)
                discriminator_output_fake = torch.sigmoid(prob_fake).squeeze(1)
                self.plot_images_with_discriminator_output(im, discriminator_output_real, real_condition)
                self.plot_images_with_discriminator_output(
                    fake_im,
                    discriminator_output_fake,
                    real_condition,
                    consistency_generated_images=True,
                )

        return d_loss

    def validation_step(self, batch, batch_idx):
        images, cond_input = batch

        # Encode images to latent space
        with torch.no_grad():
            # reconstructed, z, _ = self.vae(images, cond_input['tensor'] if cond_input else None)
            reconstructed, z, _ = self.vae(images, None)

        if self.use_noisy_images:
            t = torch.randint(0, self.scheduler.num_timesteps, (z.shape[0],)).to(self.device)
            noise = torch.randn_like(z)
            im = self.scheduler.add_noise(z, noise, t)
        else:
            im = z

        if self.disc_input_noise_std > 0:
            im = im + torch.randn_like(im) * self.disc_input_noise_std

        # Prepare real labels and fake labels with label smoothing
        real_labels = torch.full((im.size(0), 1), self.pos_label, device=self.device)
        fake_labels = torch.full((im.size(0), 1), self.neg_label, device=self.device)

        # Real condition (unchanged for real noise)
        real_condition = cond_input['tensor']

        # Generate fake conditions by rolling (shifting the tensor elements)
        fake_cond_input = {key: value.clone() for key, value in cond_input.items()}
        fake_cond_input['tensor'][:, :self.num_classes] = torch.roll(
            fake_cond_input['tensor'][:, :self.num_classes], 1,
            0
        ).long()
        fake_cond_input['tensor'][:, self.num_classes:] = torch.roll(
            fake_cond_input['tensor'][:, self.num_classes:], 1,
            0
        )
        fake_condition = fake_cond_input['tensor']

        # Generate fake images using provided generator (consistency model or other)
        with torch.no_grad():
            if self.generator is not None:
                fake_im = self.generator(fake_condition)
                if fake_im.shape != im.shape:
                    _, fake_im, _ = self.vae(fake_im, None)
            else:
                fake_im = torch.randn_like(im)

        # Step 3: Pass both real and generated fake images to the discriminator
        prob_real = self.forward(
            im,
            im,
            real_condition[:, :self.num_classes].long(),
            real_condition[:, self.num_classes:]
        )

        prob_fake = self.forward(
            fake_im,
            fake_im,
            fake_condition[:, :self.num_classes].long(),
            fake_condition[:, self.num_classes:]
        )

        # Step 4: Compute discriminator loss
        d_loss_real = self.criterion(prob_real, real_labels)
        d_loss_fake = self.criterion(prob_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2

        # Log the loss
        self.log('val/d_loss', d_loss, on_step=True, prog_bar=True)
        self.log('val/disc_real_mean', torch.sigmoid(prob_real).mean(), prog_bar=True)
        self.log('val/disc_fake_mean', torch.sigmoid(prob_fake).mean(), prog_bar=True)

    def configure_optimizers(self):
        # Configure optimizers
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay,
        )

        if not self.use_lr_scheduler:
            return optimizer

        scheduler_cfg = self.lr_scheduler_config

        warmup_epochs = scheduler_cfg.get('warmup_epochs', 5)
        warmup_start_factor = scheduler_cfg.get('warmup_start_factor', 0.1)
        eta_min = scheduler_cfg.get('eta_min', self.lr * 1e-3)
        t_mult = scheduler_cfg.get('T_mult', 1)

        # Determine the base period for cosine restarts. When the trainer is available we
        # adapt to the remaining training epochs to ensure the schedule fully decays.
        if self.trainer is not None and self.trainer.max_epochs is not None:
            total_epochs = max(self.trainer.max_epochs, warmup_epochs + 1)
            cosine_period = max(total_epochs - warmup_epochs, 1)
        else:
            cosine_period = scheduler_cfg.get('T_0', 10)

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_period,
            T_mult=t_mult,
            eta_min=eta_min,
        )

        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine_scheduler

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': scheduler_cfg.get('interval', 'epoch'),
            'frequency': scheduler_cfg.get('frequency', 1),
            'name': scheduler_cfg.get('name', 'disc_warmup_cosine'),
        }

        monitor = scheduler_cfg.get('monitor')
        if monitor is not None:
            scheduler_dict['monitor'] = monitor

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler_dict,
        }

