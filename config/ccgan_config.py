"""Configuration helpers for the LightningCcGAN training script."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from config.geometry import GeometryLabelSchema


@dataclass
class DatasetConfig:
    im_path: str
    im_size: int = 128
    im_channels: int = 1
    batch_size: int = 128
    num_workers: int = 4
    use_latents: bool = False
    latent_path: Optional[str] = None
    condition_config: Optional[Dict[str, Any]] = None
    train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    max_samples: Optional[int] = None
    geometry_label_schema: GeometryLabelSchema = field(default_factory=GeometryLabelSchema)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_label_schema, GeometryLabelSchema):
            self.geometry_label_schema = GeometryLabelSchema(**self.geometry_label_schema)


@dataclass
class ModelConfig:
    latent_dim: int = 2
    hidden_dim: int = 100
    generator_layers: int = 6
    discriminator_layers: int = 5
    radius: float = 1.0


@dataclass
class VicinalConfig:
    kernel_sigma: float = 0.05
    kappa: float = 0.1
    threshold_type: str = "hard"
    nonzero_soft_weight_threshold: float = 0.0
    circular_labels: bool = True


@dataclass
class OptimConfig:
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    batch_size_disc: int = 64
    batch_size_gene: int = 64


@dataclass
class LoggingConfig:
    wandb_project: str = "CcGANGeometry"
    wandb_entity: Optional[str] = None
    log_model: bool = True
    checkpoint_dir: str = "./checkpoints/ccgan"
    checkpoint_filename: str = "ccgan-{step}"
    checkpoint_monitor: Optional[str] = None
    checkpoint_mode: str = "min"
    save_top_k: int = 1
    enable_lr_monitor: bool = True


@dataclass
class TrainingConfig:
    max_steps: int = 5000
    log_every_n_steps: int = 100
    sample_every_n_steps: int = 1000
    sample_batch_size: int = 64


@dataclass
class CcGANExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    vicinal: VicinalConfig = field(default_factory=VicinalConfig)
    optimisation: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CcGANExperimentConfig":
        dataset_cfg = DatasetConfig(**data.get("dataset", {}))
        model_cfg = ModelConfig(**data.get("model", {}))
        vicinal_cfg = VicinalConfig(**data.get("vicinal", {}))
        optim_cfg = OptimConfig(**data.get("optimisation", {}))
        logging_cfg = LoggingConfig(**data.get("logging", {}))
        training_cfg = TrainingConfig(**data.get("training", {}))
        return cls(
            dataset=dataset_cfg,
            model=model_cfg,
            vicinal=vicinal_cfg,
            optimisation=optim_cfg,
            logging=logging_cfg,
            training=training_cfg,
        )

