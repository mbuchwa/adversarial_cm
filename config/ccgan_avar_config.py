"""Configuration helpers for the LightningCcGANAVAR training script."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config.geometry import GeometryLabelSchema


@dataclass
class DatasetConfig:
    """Parameters controlling how the geometry dataset is loaded."""

    im_path: str
    im_size: int = 64
    im_channels: int = 1
    batch_size: int = 64
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
class LabelEmbeddingConfig:
    """Pre-computed label embedding artefacts used by CcGAN-AVAR."""

    path_y2h: str
    path_y2cov: str
    y2h_type: str = "sinusoidal"
    y2cov_type: str = "sinusoidal"
    h_dim: Optional[int] = None
    cov_dim: Optional[int] = None
    batch_size: int = 128
    nc: Optional[int] = None
    device: Optional[str] = None


@dataclass
class ModelConfig:
    """Architectural hyper-parameters for the generator and discriminator."""

    net_name: str = "sngan"
    dim_z: int = 128
    dim_y: int = 128
    img_size: int = 64
    num_channels: int = 1
    gene_ch: int = 32
    disc_ch: int = 32
    ch_multi_g: Optional[List[int]] = None
    ch_multi_d: Optional[List[int]] = None
    use_sn: bool = True
    use_attn: bool = True
    use_aux_reg_branch: bool = False
    use_aux_reg_model: bool = False
    aux_reg_model_arch: str = "resnet18"
    aux_reg_model_checkpoint: Optional[str] = None
    use_dre_reg: bool = False
    dre_head_arch: str = "MLP3"


@dataclass
class VicinalConfig:
    """Vicinal sampling options for label interpolation."""

    kernel_sigma: float = 0.05
    kappa: float = 0.1
    threshold_type: str = "hard"
    nonzero_soft_weight_threshold: float = 0.0
    use_ada_vic: bool = False
    ada_vic_type: str = "standard"
    ada_eps: float = 1e-5
    min_n_per_vic: int = 1
    use_symm_vic: bool = False


@dataclass
class AuxLossConfig:
    """Auxiliary regression and density-ratio regularisers."""

    aux_reg_loss_type: str = "mse"
    aux_reg_loss_ei_hinge_factor: float = 1.0
    aux_reg_loss_huber_delta: float = 1.0
    aux_reg_loss_huber_quantile: float = 0.5
    weight_d_aux_reg_loss: float = 0.0
    weight_g_aux_reg_loss: float = 0.0
    aux_reg_checkpoint: Optional[str] = None
    dre_lambda: float = 0.0
    weight_d_aux_dre_loss: float = 0.0
    weight_g_aux_dre_loss: float = 0.0
    dre_checkpoint: Optional[str] = None


@dataclass
class OptimConfig:
    """Optimiser and gradient accumulation settings."""

    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    latent_dim: int = 128
    batch_size_disc: int = 64
    batch_size_gene: int = 64
    num_d_steps: int = 1
    num_grad_acc_d: int = 1
    num_grad_acc_g: int = 1
    max_grad_norm: float = 1.0
    betas: Tuple[float, float] = (0.5, 0.999)


@dataclass
class LoggingConfig:
    """Weights & Biases project setup and checkpoint schedule."""

    wandb_project: str = "CcGANAVAR"
    wandb_entity: Optional[str] = None
    log_model: bool = True
    checkpoint_dir: str = "./checkpoints/ccgan-avar"
    checkpoint_filename: str = "ccgan-avar-{step}"
    checkpoint_monitor: Optional[str] = None
    checkpoint_mode: str = "min"
    save_top_k: int = 1
    enable_lr_monitor: bool = True


@dataclass
class EMAConfig:
    """Exponential moving average tracking for the generator."""

    enabled: bool = False
    update_after_step: int = int(1e30)
    update_every: int = 10
    decay: float = 0.999


@dataclass
class DiffAugConfig:
    """Differentiable augmentation policy applied to discriminator inputs."""

    enabled: bool = False
    policy: str = "color,translation,cutout"


@dataclass
class TrainingConfig:
    """Global training schedule and runtime toggles."""

    max_steps: int = 50000
    log_every_n_steps: int = 100
    sample_every_n_steps: int = 1000
    sample_batch_size: int = 64
    save_images_dir: Optional[str] = None
    checkpoint_sampling_freq: int = 0
    use_amp: bool = False
    mixed_precision_type: str = "fp16"
    loss_type: str = "hinge"
    ema: EMAConfig = field(default_factory=EMAConfig)
    diffaug: DiffAugConfig = field(default_factory=DiffAugConfig)


@dataclass
class CcGANAVARExperimentConfig:
    """Bundle of configuration blocks consumed by the AVAR training script."""

    dataset: DatasetConfig
    label_embedding: LabelEmbeddingConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    vicinal: VicinalConfig = field(default_factory=VicinalConfig)
    aux_loss: AuxLossConfig = field(default_factory=AuxLossConfig)
    optimisation: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CcGANAVARExperimentConfig":
        dataset_cfg = DatasetConfig(**data.get("dataset", {}))
        label_embed_cfg = LabelEmbeddingConfig(**data.get("label_embedding", {}))
        model_cfg = ModelConfig(**data.get("model", {}))
        vicinal_cfg = VicinalConfig(**data.get("vicinal", {}))
        aux_loss_cfg = AuxLossConfig(**data.get("aux_loss", {}))
        optim_cfg = OptimConfig(**data.get("optimisation", {}))
        logging_cfg = LoggingConfig(**data.get("logging", {}))
        training_data = data.get("training") or {}
        ema_cfg = EMAConfig(**training_data.get("ema", {}))
        diffaug_cfg = DiffAugConfig(**training_data.get("diffaug", {}))
        training_cfg = TrainingConfig(
            ema=ema_cfg,
            diffaug=diffaug_cfg,
            **{k: v for k, v in training_data.items() if k not in {"ema", "diffaug"}},
        )
        return cls(
            dataset=dataset_cfg,
            label_embedding=label_embed_cfg,
            model=model_cfg,
            vicinal=vicinal_cfg,
            aux_loss=aux_loss_cfg,
            optimisation=optim_cfg,
            logging=logging_cfg,
            training=training_cfg,
        )

