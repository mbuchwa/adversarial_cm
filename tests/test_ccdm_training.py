"""Tests for the CCDM training script configuration helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from ccdm_training import _instantiate_module


def test_instantiate_module_from_default_config(tmp_path) -> None:
    config_path = Path("config/ccdm_geometries.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Redirect artefacts to the temporary directory so tests remain hermetic.
    ccdm_cfg = config.setdefault("ccdm_params", {})
    ccdm_cfg["results_folder"] = str(tmp_path / "results")
    label_embed_cfg = config.setdefault("label_embedding_params", {})
    label_embed_cfg["path_y2h"] = str(tmp_path / "y2h")
    label_embed_cfg["path_y2cov"] = str(tmp_path / "y2cov")

    module = _instantiate_module(config)

    dataset_cfg = config["dataset_params"]
    sampling_cfg = config.get("sampling_params", {})
    vicinal_cfg = config.get("vicinal_params", {})
    diffusion_cfg = config.get("diffusion_params", {})

    assert module.lr == ccdm_cfg.get("train_lr", module.lr)

    assert module.image_size == dataset_cfg["im_size"]
    assert module.in_channels == dataset_cfg["im_channels"]
    assert module.sample_batch_size == sampling_cfg.get("sample_batch_size", module.sample_batch_size)
    assert module.cond_scale == sampling_cfg.get("guidance_scale", module.cond_scale)
    assert module.sample_every_n_steps == sampling_cfg.get("log_every_n_training_steps", module.sample_every_n_steps)
    assert module.vicinal_params.kernel_sigma == vicinal_cfg.get("kernel_sigma", module.vicinal_params.kernel_sigma)
    assert module.vicinal_params.kappa == vicinal_cfg.get("kappa", module.vicinal_params.kappa)
    if "pred_objective" in diffusion_cfg:
        assert module.diffusion.objective == diffusion_cfg["pred_objective"]
    if "sample_timesteps" in diffusion_cfg:
        assert module.diffusion.sampling_timesteps == diffusion_cfg["sample_timesteps"]
