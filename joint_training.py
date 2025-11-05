import argparse
import os
import yaml
import pytorch_lightning as pl
import torch
import numpy as np
import random
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.pl_joint_5 import JointDiffusionTrainer
from models.unet_cond import Unet
from models.pl_vqvae import LightningVQVAE
from models.pl_discriminator import LightningConditionalDiscriminator
from models.pl_consistency_model import LightningConsistencyModel
from denoiser.karras_denoiser import KarrasDiffusion
from denoiser.linear_noise_scheduler import LinearNoiseScheduler
from utils.resample import create_named_schedule_sampler
from utils.config_utils import get_config_value
from dataset.geometries_dataset import GeometriesDataModule
from pytorch_lightning.utilities import rank_zero_only

torch.set_float32_matmul_precision('medium')


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # These settings ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    pl.seed_everything(seed, workers=True)  # Also sets seed for dataloader workers


@rank_zero_only
def log_config_to_wandb(wandb_logger, config, seed):
    wandb_logger.experiment.config.update(config)
    wandb_logger.experiment.config.update({'random_seed': seed})


@rank_zero_only
def log_consistency_learning_rate(wandb_logger, learning_rate):
    """Log the consistency learning rate to stdout and Weights & Biases."""
    if wandb_logger is not None:
        wandb_logger.log_metrics({'train/consistency_learning_rate': learning_rate}, step=0)
    print(f"Consistency learning rate: {learning_rate}")


def train(args, wandb_run=None):
    # Set random seed first thing
    set_seed(args.seed)

    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    diffusion_model_config = config['ldm_params']
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    consistency_config = config['consistency_params']
    discriminator_config = config['discriminator_params']

    data_module = GeometriesDataModule(
        data_dir=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        batch_size=train_config['ldm_batch_size'],
        num_workers=16,
        use_latents=True,
        latent_path=os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name']),
        condition_config=get_config_value(diffusion_model_config, 'condition_config', None)
    )

    # Load the checkpoint if specified
    vqvae_checkpoint_path = train_config.get('vqvae_autoencoder_ckpt_name', None)
    if vqvae_checkpoint_path:
        vae = LightningVQVAE.load_from_checkpoint(
            vqvae_checkpoint_path,
            im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config
        )
        vae.eval()
    else:
        raise ValueError("VAE path must be specified for latent diffusion setup!")

    #################################################
    # LDM and consistency setup
    #################################################

    # Initialize models and scheduler
    consistency_model_config = diffusion_model_config
    consistency_model_config['dropout_rate'] = 0.0
    unet_consistency = Unet(autoencoder_model_config['z_channels'], consistency_model_config)
    unet_consistency.train()

    # Initialize models and scheduler
    unet_target = Unet(autoencoder_model_config['z_channels'], consistency_model_config)
    unet_target.train()

    schedule_sampler = create_named_schedule_sampler('uniform', diffusion_config["num_timesteps"])


    # Initialize noise scheduler
    karras_diffusion_consistency = KarrasDiffusion(
        sigma_min=consistency_config["sigma_min"],
        sigma_max=consistency_config["sigma_max"],
        rho=consistency_config["rho"],
        distillation=False,
        steps=consistency_config["num_scales"],  # 1
        loss_norm=consistency_config["loss_norm"],
        weight_schedule=consistency_config["weight_schedule"]
    )

    # Initialize LightningConsistencyModel
    consistency_checkpoint_path = train_config.get('consistency_ckpt_cond_name', None)
    if consistency_checkpoint_path is not None and use_pretrained_consistency:
        consistency_model = LightningConsistencyModel.load_from_checkpoint(
            consistency_checkpoint_path,
            model=unet_consistency,
            diffusion=karras_diffusion_consistency,
            vae_model=vae,
            teacher_model=None,
            target_model=unet_target,
            schedule_sampler=schedule_sampler,
            teacher_diffusion=None,
            training_mode="consistency_training",
            plot_example_images_epoch_start=10,
            lr=train_config['consistency_lr'],
            num_scales=consistency_config["num_scales"],
            weight_decay=consistency_config["weight_decay"],
            ema_decay=consistency_config["ema_decay"],
            sigma_min=diffusion_config["sigma_min"],
            sigma_max=diffusion_config["sigma_max"],
            rho=diffusion_config["rho"]
        )
        # Manually restore EMA weights from the checkpoint
        checkpoint = torch.load(consistency_checkpoint_path, map_location=device)

        if "ema" in checkpoint["state_dict"]:
            consistency_model.ema.load_state_dict(checkpoint["state_dict"]["ema"])
            print("✅ EMA weights restored from checkpoint!")
        else:
            print("⚠️ Warning: No EMA weights found in checkpoint!")

    else:
        consistency_model = LightningConsistencyModel(
            model=unet_consistency,
            diffusion=karras_diffusion_consistency,
            vae_model=vae,
            teacher_model=None,
            target_model=unet_target,
            schedule_sampler=schedule_sampler,
            teacher_diffusion=None,
            training_mode="consistency_training",
            plot_example_images_epoch_start=10,
            lr=train_config['consistency_lr'],
            num_scales=consistency_config["num_scales"],
            weight_decay=consistency_config["weight_decay"],
            ema_decay=consistency_config["ema_decay"],
            sigma_min=diffusion_config["sigma_min"],
            sigma_max=diffusion_config["sigma_max"],
            rho=diffusion_config["rho"]
        ).to(device)

    consistency_model.ema.to(device)

    consistency_model.train()

    #################################################
    # Discriminator setup
    #################################################

    noise_scheduler = LinearNoiseScheduler(diffusion_config['num_timesteps'],
                                           diffusion_config['beta_start'],
                                           diffusion_config['beta_end'])

    """Load the Discriminator checkpoint if specified"""
    disc_checkpoint_path = train_config.get('vqvae_discriminator_ckpt_name', None)
    if disc_checkpoint_path is not None and use_pretrained_discriminator:
        discriminator = LightningConditionalDiscriminator(
            vae_model=vae,
            scheduler=noise_scheduler,
            image_channels=discriminator_config['image_channels'],
            num_classes=diffusion_model_config['condition_config']['tensor_condition_config']['num_class_variables'],
            num_continuous_features=diffusion_model_config['condition_config']['tensor_condition_config']['num_cont_features'],
            num_objects=diffusion_model_config['condition_config']['tensor_condition_config']['num_objects'],
            embedding_dim=discriminator_config['embedding_dim'],
            num_blocks=discriminator_config['num_blocks'],
            lr=train_config['disc_lr'],
            initial_features=discriminator_config['initial_features']
        ).to(device)

        checkpoint = torch.load(disc_checkpoint_path, map_location=device)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        print(f"✅ Discriminator loaded from {disc_checkpoint_path}")
    else:
        discriminator = LightningConditionalDiscriminator(
            vae_model=vae,
            scheduler=noise_scheduler,
            image_channels=discriminator_config['image_channels'],
            num_classes=diffusion_model_config['condition_config']['tensor_condition_config']['num_class_variables'],
            num_continuous_features=diffusion_model_config['condition_config']['tensor_condition_config']['num_cont_features'],
            num_objects=diffusion_model_config['condition_config']['tensor_condition_config']['num_objects'],
            embedding_dim=discriminator_config['embedding_dim'],
            num_blocks=discriminator_config['num_blocks'],
            num_scales=consistency_config['num_scales'],
            lr=train_config['disc_lr'],
            initial_features=discriminator_config['initial_features'],
            pos_label=1.0 - train_config.get('label_smoothing', 0.1),
            neg_label=train_config.get('label_smoothing', 0.1)
        ).to(device)
    discriminator.train()

    """Init the Joint Model checkpoint"""
    joint_model = JointDiffusionTrainer(
        consistency_model=consistency_model,
        discriminator=discriminator,
        vae=vae,
        batch_size=train_config['ldm_batch_size'],
        learning_rate=train_config['consistency_lr'],
        learining_rate_consistency=train_config['consistency_lr'],
        learning_rate_disc=train_config['disc_lr'],
        use_discriminator=True,
        weight_decay_disc=discriminator_config.get('weight_decay', 0.0),
        discriminator_start_epoch=train_config['discriminator_start_epoch'],
        discriminator_weight=train_config['discriminator_weight'],
        discriminator_steps=train_config.get('discriminator_steps', 1),
        disc_loss_type=train_config.get('disc_loss_type', 'bce'),
        disc_input_noise_std=train_config.get('disc_input_noise_std', 0.0),
        label_smoothing=train_config.get('label_smoothing', 0.1),
        max_discriminator_steps=train_config.get('max_discriminator_steps', 5),
        gradient_penalty_weight=train_config.get('gradient_penalty_weight', 10.0),
        adv_ramp_up_epochs=train_config.get('adv_ramp_up_epochs', 50),
        freeze_cm_epoch=train_config.get('freeze_cm_epoch', 0),
        lambda_adv=train_config.get('lambda_adv', 0.1),
        lambda_adv_min=train_config.get('lambda_adv_min', 0.1),
        lambda_adv_max=train_config.get('lambda_adv_max', 2.0),
        num_classes=diffusion_model_config['condition_config']['tensor_condition_config']['num_class_variables'],
        num_objects=diffusion_model_config['condition_config']['tensor_condition_config']['num_objects'],
        num_cond_variables=diffusion_model_config['condition_config']['tensor_condition_config']['num_cond_variables'],
        sigma_min=diffusion_config["sigma_min"],
        sigma_max=diffusion_config["sigma_max"],
        rho=diffusion_config["rho"],
        rollout_schedule=train_config.get('rollout_schedule'),
        rollout_warmup_epochs=train_config.get('rollout_warmup_epochs'),
        rollout_plateau_patience=train_config.get('rollout_plateau_patience'),
        rollout_plateau_delta=train_config.get('rollout_plateau_delta'),
        rollout_plateau_cooldown=train_config.get('rollout_plateau_cooldown'),
        max_rollout=10,  # consistency_config.get('num_scales'),  # set max rollout value to # of all cm steps (tunable)
        min_rollout=1
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/joint_model/',
        filename='consistency-{epoch:02d}-{train/consistency_loss:.4f}',
        save_last=True,
        save_top_k=0,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/consistency_loss',
        patience=1000,
        mode='min',
        verbose=True
    )

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "JointConsistencyExperiment"),
        entity=os.environ.get("WANDB_ENTITY"),
        log_model=True,
        experiment=wandb_run
    )

    log_config_to_wandb(wandb_logger, config, args.seed)
    log_consistency_learning_rate(wandb_logger, train_config['consistency_lr'])

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        precision='16-mixed'  # Enable mixed precision (16-bit floating point)
    )

    trainer.fit(joint_model, data_module)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    use_pretrained_consistency = False
    use_pretrained_discriminator = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='./config/geometries_cond.yaml', type=str)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    train(args)
