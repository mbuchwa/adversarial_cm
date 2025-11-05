import yaml
import os
import pytorch_lightning as pl
import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataset.geometries_dataset import GeometriesDataModule  # Your dataset
from models.pl_vqvae import LightningVQVAE  # Your Lightning VQ-VAE wrapper


def train(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_params']
    vqvae_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Initialize DataModule
    data_module = GeometriesDataModule(
        data_dir=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        batch_size=train_config['autoencoder_batch_size'],
        num_workers=4  # Adjust based on your hardware
    )

    # Initialize VQ-VAE model
    vqvae = LightningVQVAE(
        im_channels=dataset_config['im_channels'],
        model_config=vqvae_model_config,
        learning_rate=train_config['autoencoder_lr'],
        sample_log_interval=5
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/vqvae/',
        filename='vqvae-{epoch:02d}-{train/total_loss:.4f}',
        save_last=True,  # Save only the last model
        save_top_k=0,  # Keep only one checkpoint (the best one)
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/total_loss',  # Metric to monitor
        patience=train_config['patience'],  # Number of epochs with no improvement after which training will be stopped
        mode='min',  # 'min' because we want to minimize the loss
        verbose=True
    )

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project="VQVAEExperiment",  # Set your project name
        log_model=True  # Logs the model checkpoints if enabled
    )
    wandb_logger.experiment.config.update(config)  # Log the config to WandB

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=train_config['autoencoder_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],  # Add EarlyStopping to callbacks
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision=16  # Enable mixed precision (16-bit floating point)
    )

    # Train model
    trainer.fit(vqvae, data_module)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='./config/geometries_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)

