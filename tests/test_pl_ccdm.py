import numpy as np
import torch
from unittest.mock import MagicMock, PropertyMock, patch

from models.pl_ccdm import LightningCCDM, VicinalParams


def _lightweight_module(tmp_path, train_images, train_labels, **kwargs) -> LightningCCDM:
    return LightningCCDM(
        image_size=train_images.shape[-1],
        in_channels=train_images.shape[1],
        train_images=train_images,
        train_labels=train_labels,
        vicinal_params=kwargs.get(
            "vicinal_params",
            VicinalParams(kernel_sigma=0.0, kappa=0.0),
        ),
        unet_kwargs=dict(
            dim=8,
            embed_input_dim=12,
            dim_mults=(1, 2),
            in_channels=train_images.shape[1],
            attn_dim_head=4,
            attn_heads=1,
            cond_drop_prob=0.0,
        ),
        diffusion_kwargs=dict(
            timesteps=10,
            objective="pred_noise",
            use_Hy=False,
        ),
        label_embed_kwargs=dict(
            h_dim=12,
            cov_dim=24,
            y2cov_type="gaussian",
            device="cpu",
        ),
        results_folder=str(tmp_path / "results"),
        sample_every_n_steps=0,
        **kwargs,
    )


def test_training_step_accepts_vector_labels(tmp_path):
    rng = np.random.RandomState(0)
    train_images = rng.randn(16, 1, 8, 8).astype(np.float32)
    train_labels = rng.rand(16, 3).astype(np.float32)

    module = _lightweight_module(tmp_path, train_images, train_labels)
    module.train()

    batch_images = torch.from_numpy(train_images[:4])
    batch_labels = torch.from_numpy(train_labels[:4])

    loss = module.training_step((batch_images, {"tensor": batch_labels}), batch_idx=0)
    assert loss.shape == ()


def test_vicinal_sampling_soft_weights_vector_labels(tmp_path):
    train_images = np.zeros((4, 1, 4, 4), dtype=np.float32)
    train_labels = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    vicinal = VicinalParams(
        kernel_sigma=0.0,
        kappa=1.0,
        threshold_type="soft",
        nonzero_soft_weight_threshold=1e-2,
    )

    module = _lightweight_module(tmp_path, train_images, train_labels, vicinal_params=vicinal)
    module.train()

    images, labels, weights = module._sample_training_batch(batch_size=2, device=torch.device("cpu"))

    assert labels.shape == (2, 2)
    assert weights is not None
    assert weights.shape == (2,)
    assert torch.all(weights > 0)


def test_optimizer_step_triggers_sampling(tmp_path):
    rng = np.random.RandomState(0)
    train_images = rng.randn(8, 1, 8, 8).astype(np.float32)
    train_labels = rng.rand(8, 2).astype(np.float32)

    module = _lightweight_module(tmp_path, train_images, train_labels, sample_every_n_steps=2)
    module.trainer = object()
    module._log_sample_grid = MagicMock()
    module.ema.update = MagicMock()

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    with patch.object(
        LightningCCDM.__bases__[0],
        "optimizer_step",
        return_value=None,
    ), patch.object(
        type(module),
        "global_step",
        new_callable=PropertyMock,
        return_value=2,
    ):
        module.optimizer_step(0, 0, optimizer)

    module._log_sample_grid.assert_called_once_with(2)
    assert module._last_sample_step == 2


def test_epoch_end_sampling_respects_cadence(tmp_path):
    rng = np.random.RandomState(1)
    train_images = rng.randn(8, 1, 8, 8).astype(np.float32)
    train_labels = rng.rand(8, 2).astype(np.float32)

    module = _lightweight_module(tmp_path, train_images, train_labels, sample_every_n_steps=100)
    module.trainer = object()
    module._visual_labels = torch.zeros(1, 2)
    module._log_sample_grid = MagicMock()

    with patch.object(
        type(module),
        "global_step",
        new_callable=PropertyMock,
        return_value=5,
    ), patch.object(
        type(module),
        "current_epoch",
        new_callable=PropertyMock,
        return_value=1,
    ):
        module.sample_every_n_epochs = 2
        module.on_train_epoch_end()

    module._log_sample_grid.assert_called_once_with(5)
    assert module._last_sample_step == 5


def test_epoch_end_skip_when_step_already_logged(tmp_path):
    rng = np.random.RandomState(2)
    train_images = rng.randn(8, 1, 8, 8).astype(np.float32)
    train_labels = rng.rand(8, 2).astype(np.float32)

    module = _lightweight_module(tmp_path, train_images, train_labels, sample_every_n_steps=2)
    module.trainer = object()
    module._visual_labels = torch.zeros(1, 2)
    module._log_sample_grid = MagicMock()
    module._last_sample_step = 4

    with patch.object(
        type(module),
        "global_step",
        new_callable=PropertyMock,
        return_value=4,
    ), patch.object(
        type(module),
        "current_epoch",
        new_callable=PropertyMock,
        return_value=0,
    ):
        module.sample_every_n_epochs = 1
        module.on_train_epoch_end()

    module._log_sample_grid.assert_not_called()

