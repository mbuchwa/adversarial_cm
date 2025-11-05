import math
from typing import Optional
from unittest.mock import ANY, MagicMock

import pytest

torch = pytest.importorskip("torch")

from config.geometry import GeometryLabelSchema
from models.ccgan_networks import (
    ContinuousConditionalDiscriminator,
    ContinuousConditionalGenerator,
    ContinuousDiscriminatorConfig,
    ContinuousGeneratorConfig,
)
from torch.nn import functional as F
from models.pl_ccgan_avar import AVAROptimisationConfig, LightningCcGANAVAR


def _build_module(
    label_dim: int,
    distance_norm: str = "l2",
    kernel_sigma: float = 0.0,
    geometry_label_schema: Optional[GeometryLabelSchema] = None,
) -> LightningCcGANAVAR:
    schema = geometry_label_schema or GeometryLabelSchema()
    sample_dim = 4
    num_train = 8
    train_samples = torch.rand(num_train, sample_dim)
    train_labels = torch.rand(num_train, label_dim)
    eval_labels = torch.rand(2, label_dim)

    generator = ContinuousConditionalGenerator(
        ContinuousGeneratorConfig(
            latent_dim=2,
            output_dim=sample_dim,
            hidden_dim=8,
            num_hidden_layers=1,
            label_dim=label_dim,
        )
    )
    discriminator = ContinuousConditionalDiscriminator(
        ContinuousDiscriminatorConfig(
            input_dim=sample_dim,
            hidden_dim=8,
            num_hidden_layers=1,
            label_dim=label_dim,
        )
    )

    vicinal_params = {
        "use_ada_vic": False,
        "kappa": 0.25,
        "threshold_type": "hard",
        "distance_norm": distance_norm,
    }
    if kernel_sigma is not None:
        vicinal_params["kernel_sigma"] = kernel_sigma
    aux_loss_params = {
        "aux_reg_loss_type": "mse",
        "weight_d_aux_reg_loss": 0.0,
        "weight_g_aux_reg_loss": 0.0,
        "use_aux_reg_branch": False,
        "use_aux_reg_model": False,
        "use_dre_reg": False,
        "weight_d_aux_dre_loss": 0.0,
        "weight_g_aux_dre_loss": 0.0,
    }

    optimisation_config = AVAROptimisationConfig(
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        latent_dim=2,
        batch_size_disc=2,
        batch_size_gene=2,
        num_d_steps=1,
        num_grad_acc_d=1,
        num_grad_acc_g=1,
        max_grad_norm=math.inf,
    )

    module = LightningCcGANAVAR(
        train_samples=train_samples,
        train_labels=train_labels,
        eval_labels=eval_labels,
        generator=generator,
        discriminator=discriminator,
        fn_y2h=lambda labels: labels,
        vicinal_params=vicinal_params,
        aux_loss_params=aux_loss_params,
        optimisation_config=optimisation_config,
        sample_shape=(sample_dim,),
        geometry_label_schema=schema,
    )
    return module


def test_on_validation_epoch_end_logs_metrics():
    module = _build_module(label_dim=1)
    module.sample_shape = (1, 2, 2)

    sample_calls = []

    def fake_sample(labels: torch.Tensor, *, use_ema: bool = False):
        sample_calls.append(use_ema)
        batch = labels.size(0)
        return torch.zeros(batch, *module.sample_shape)

    module.sample = fake_sample  # type: ignore[assignment]

    def fake_compute(labels, samples, *, prefix="train", metric_tag=None):
        tag = metric_tag or "sample"
        base = f"{prefix}/{tag}_sample"
        return (
            {
                f"{base}_mse_mean": 1.0,
                f"{base}_ssim_mean": 0.5,
                f"{base}_psnr_mean": 2.0,
                f"{base}_lpips_mean": 0.1,
            },
            None,
        )

    module._compute_sampling_metrics = MagicMock(side_effect=fake_compute)  # type: ignore[assignment]

    logged = []

    def fake_log(key, value, **kwargs):
        logged.append((key, value, kwargs))

    module.log = fake_log  # type: ignore[assignment]
    module.logger = None
    module.ema_g = object()

    module.on_validation_epoch_end()

    assert sample_calls == [False, True]

    module._compute_sampling_metrics.assert_any_call(ANY, ANY, prefix="val", metric_tag="generator")
    module._compute_sampling_metrics.assert_any_call(
        ANY, ANY, prefix="val", metric_tag="ema_generator"
    )

    logged_keys = {key for key, _, _ in logged}
    expected_keys = {
        "val/generator_sample_mse_mean",
        "val/generator_sample_ssim_mean",
        "val/generator_sample_psnr_mean",
        "val/generator_sample_lpips_mean",
        "val/ema_generator_sample_mse_mean",
        "val/ema_generator_sample_ssim_mean",
        "val/ema_generator_sample_psnr_mean",
        "val/ema_generator_sample_lpips_mean",
    }
    assert expected_keys.issubset(logged_keys)

    for key, _, kwargs in logged:
        if key in expected_keys:
            assert kwargs.get("on_step") is False
            assert kwargs.get("on_epoch") is True


def test_make_vicinity_vector_labels_returns_valid_shapes():
    module = _build_module(label_dim=3)
    raw = module._sample_raw_labels(4)
    targets = module._sample_target_labels(raw)

    (
        real_indices,
        fake_labels,
        real_labels,
        real_weights,
        fake_weights,
        kappa_l,
        kappa_r,
    ) = module._make_vicinity(targets, raw)

    assert real_indices.shape == (4,)
    assert fake_labels.shape == (4, 3)
    assert real_labels.shape == (4, 3)
    assert torch.all(fake_labels >= 0.0) and torch.all(fake_labels <= 1.0)
    assert torch.all(real_labels >= 0.0) and torch.all(real_labels <= 1.0)
    assert real_weights.shape == (4,)
    assert fake_weights.shape == (4,)
    assert kappa_l.shape == (4,)
    assert kappa_r.shape == (4,)


def test_make_vicinity_supports_alternative_norms():
    module = _build_module(label_dim=2, distance_norm="l1")
    raw = module._sample_raw_labels(3)
    targets = module._sample_target_labels(raw)

    _, _, _, real_weights, fake_weights, _, _ = module._make_vicinity(targets, raw)

    assert real_weights.shape == (3,)
    assert fake_weights.shape == (3,)


def test_sample_target_labels_respects_geometry_schema():
    label_dim = 8
    sample_dim = 4
    num_train = 16

    shape_indices = torch.randint(0, 4, (num_train,))
    one_hot = F.one_hot(shape_indices, num_classes=4).float()
    positions = torch.rand(num_train, 2) * 0.6 + 0.2
    sizes = torch.rand(num_train, 1) * 0.1 + 0.1
    rotations = torch.rand(num_train, 1)
    continuous = torch.cat([positions, sizes, rotations], dim=1)

    train_samples = torch.rand(num_train, sample_dim)
    train_labels = torch.cat([one_hot, continuous], dim=1)
    eval_labels = train_labels[:2]

    schema = GeometryLabelSchema()

    generator = ContinuousConditionalGenerator(
        ContinuousGeneratorConfig(
            latent_dim=2,
            output_dim=sample_dim,
            hidden_dim=8,
            num_hidden_layers=1,
            label_dim=label_dim,
        )
    )
    discriminator = ContinuousConditionalDiscriminator(
        ContinuousDiscriminatorConfig(
            input_dim=sample_dim,
            hidden_dim=8,
            num_hidden_layers=1,
            label_dim=label_dim,
        )
    )

    module = LightningCcGANAVAR(
        train_samples=train_samples,
        train_labels=train_labels,
        eval_labels=eval_labels,
        generator=generator,
        discriminator=discriminator,
        fn_y2h=lambda labels: labels,
        vicinal_params={
            "use_ada_vic": False,
            "kappa": 0.25,
            "threshold_type": "hard",
            "distance_norm": "l2",
            "kernel_sigma": 0.2,
        },
        aux_loss_params={
            "aux_reg_loss_type": "mse",
            "weight_d_aux_reg_loss": 0.0,
            "weight_g_aux_reg_loss": 0.0,
            "use_aux_reg_branch": False,
            "use_aux_reg_model": False,
            "use_dre_reg": False,
            "weight_d_aux_dre_loss": 0.0,
            "weight_g_aux_dre_loss": 0.0,
        },
        optimisation_config=AVAROptimisationConfig(
            generator_lr=1e-4,
            discriminator_lr=1e-4,
            latent_dim=2,
            batch_size_disc=2,
            batch_size_gene=2,
            num_d_steps=1,
            num_grad_acc_d=1,
            num_grad_acc_g=1,
            max_grad_norm=math.inf,
        ),
        sample_shape=(sample_dim,),
        geometry_label_schema=schema,
    )

    raw = module._sample_raw_labels(6)
    targets = module._sample_target_labels(raw)

    one_hot_dim = schema.one_hot_dim
    assert torch.allclose(
        targets[:, :one_hot_dim].sum(dim=1), torch.ones(6, device=targets.device)
    )
    assert torch.all((targets[:, :one_hot_dim] == 0.0) | (targets[:, :one_hot_dim] == 1.0))

    cont_slice = slice(one_hot_dim, one_hot_dim + len(schema.continuous_order))
    cont = targets[:, cont_slice]
    bounds = [schema.continuous_bounds[key] for key in schema.continuous_order]
    lower = torch.tensor([bound[0] for bound in bounds], device=cont.device)
    upper = torch.tensor([bound[1] for bound in bounds], device=cont.device)
    assert torch.all(cont >= lower)
    assert torch.all(cont <= upper)


def test_label_embedding_dimension_validation():
    sample_dim = 3
    train_samples = torch.rand(6, sample_dim)
    labels = torch.rand(6, 2)
    eval_labels = torch.rand(2, 2)

    generator = ContinuousConditionalGenerator(
        ContinuousGeneratorConfig(
            latent_dim=2,
            output_dim=sample_dim,
            hidden_dim=8,
            num_hidden_layers=1,
            label_dim=2,
        )
    )
    discriminator = ContinuousConditionalDiscriminator(
        ContinuousDiscriminatorConfig(
            input_dim=sample_dim,
            hidden_dim=8,
            num_hidden_layers=1,
            label_dim=2,
        )
    )

    vicinal_params = {"use_ada_vic": False, "kappa": 0.1, "threshold_type": "hard"}
    aux_loss_params = {
        "aux_reg_loss_type": "mse",
        "weight_d_aux_reg_loss": 0.0,
        "weight_g_aux_reg_loss": 0.0,
        "use_aux_reg_branch": False,
        "use_aux_reg_model": False,
        "use_dre_reg": False,
        "weight_d_aux_dre_loss": 0.0,
        "weight_g_aux_dre_loss": 0.0,
    }

    optimisation_config = AVAROptimisationConfig(
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        latent_dim=2,
        batch_size_disc=2,
        batch_size_gene=2,
    )

    def mismatched_fn(labels: torch.Tensor) -> torch.Tensor:
        return torch.cat([labels, labels], dim=1)

    with pytest.raises(ValueError, match="Generator label embedding dimensionality"):
        LightningCcGANAVAR(
            train_samples=train_samples,
            train_labels=labels,
            eval_labels=eval_labels,
            generator=generator,
            discriminator=discriminator,
            fn_y2h=mismatched_fn,
            vicinal_params=vicinal_params,
            aux_loss_params=aux_loss_params,
            optimisation_config=optimisation_config,
            sample_shape=(sample_dim,),
        )
