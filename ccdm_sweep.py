"""Utility to launch Weights & Biases sweeps for the CCDM trainer."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml

import ccdm_training

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

import wandb


PARAMETER_PATHS: Mapping[str, Sequence[str]] = {
    "ccdm.train_lr": ("ccdm_params", "train_lr"),
    "ccdm.train_batch_size": ("ccdm_params", "train_batch_size"),
    "ccdm.gradient_accumulate_every": ("ccdm_params", "gradient_accumulate_every"),
    "ccdm.ema_decay": ("ccdm_params", "ema_decay"),
    "ccdm.ema_update_every": ("ccdm_params", "ema_update_every"),
    "unet.model_channels": ("unet_params", "model_channels"),
    "unet.num_res_blocks": ("unet_params", "num_res_blocks"),
    "diffusion.train_timesteps": ("diffusion_params", "train_timesteps"),
    "diffusion.sample_timesteps": ("diffusion_params", "sample_timesteps"),
    "trainer.max_steps": ("trainer_params", "max_steps"),
}


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _safe_factor_values(value: float, factors: Iterable[float]) -> Sequence[float]:
    values = {round(float(value) * float(factor), 8) for factor in factors}
    values.add(round(float(value), 8))
    values = {v for v in values if v > 0}
    return sorted(values)


def _build_parameter_values(config: Mapping[str, Any]) -> Dict[str, Any]:
    ccdm_cfg = config.get("ccdm_params", {})
    unet_cfg = config.get("unet_params", {})
    diffusion_cfg = config.get("diffusion_params", {})
    trainer_cfg = config.get("trainer_params", {})

    train_lr = float(ccdm_cfg.get("train_lr", 1e-4))
    train_batch_size = int(ccdm_cfg.get("train_batch_size", 64))
    grad_accum = int(ccdm_cfg.get("gradient_accumulate_every", 1))
    ema_decay = float(ccdm_cfg.get("ema_decay", 0.995))
    ema_update_every = int(ccdm_cfg.get("ema_update_every", 10))
    model_channels = int(unet_cfg.get("model_channels", 128))
    num_res_blocks = int(unet_cfg.get("num_res_blocks", 2))
    train_timesteps = int(diffusion_cfg.get("train_timesteps", 1000))
    sample_timesteps = int(diffusion_cfg.get("sample_timesteps", 50))
    max_steps = int(trainer_cfg.get("max_steps", 200_000))

    parameter_values: Dict[str, Any] = {
        "ccdm.train_lr": {"values": _safe_factor_values(train_lr, (0.5, 1.0, 2.0))},
        "ccdm.train_batch_size": {"values": sorted({32, train_batch_size, 64, 96})},
        "ccdm.gradient_accumulate_every": {"values": sorted({1, grad_accum, 2, 3})},
        "ccdm.ema_decay": {"values": sorted({round(v, 5) for v in (0.99, ema_decay, 0.9995) if v < 1.0})},
        "ccdm.ema_update_every": {"values": sorted({5, ema_update_every, 10, 20})},
        "unet.model_channels": {"values": sorted({96, model_channels, 128, 160})},
        "unet.num_res_blocks": {"values": sorted({2, num_res_blocks, 3, 4})},
        "diffusion.train_timesteps": {"values": sorted({750, train_timesteps, 1000, 1250})},
        "diffusion.sample_timesteps": {"values": sorted({30, sample_timesteps, 50, 70})},
        "trainer.max_steps": {"values": sorted({150_000, max_steps, 200_000, 250_000})},
    }

    for key, value_dict in parameter_values.items():
        values = value_dict["values"]
        if not values:
            raise ValueError(f"Parameter '{key}' must define at least one value for the sweep.")
    return parameter_values


def build_sweep_config(config: Mapping[str, Any], method: str, sweep_name: str | None = None) -> Dict[str, Any]:
    parameter_values = _build_parameter_values(config)
    sweep_config: Dict[str, Any] = {
        "name": sweep_name or "ccdm_sweep",
        "method": method,
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": parameter_values,
    }
    return sweep_config


def _set_nested(config: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    cursor: MutableMapping[str, Any] = config
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[path[-1]] = value


def apply_sweep_config(base_config: Mapping[str, Any], sweep_config: Mapping[str, Any]) -> Dict[str, Any]:
    updated_config = copy.deepcopy(base_config)
    for key, value in sweep_config.items():
        if key not in PARAMETER_PATHS or key.startswith("_"):
            continue
        _set_nested(updated_config, PARAMETER_PATHS[key], value)
    return updated_config


def _prepare_wandb_env(project: str | None, entity: str | None) -> None:
    if project:
        os.environ.setdefault("WANDB_PROJECT", project)
    if entity:
        os.environ.setdefault("WANDB_ENTITY", entity)


def run_training_with_config(config: Dict[str, Any], seed: int, wandb_run: Run | None) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(config, tmp)
        tmp_path = tmp.name
    try:
        args = argparse.Namespace(config_path=tmp_path, seed=seed)
        ccdm_training.train(args, wandb_run=wandb_run)
    finally:
        os.remove(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Weights & Biases sweep for the CCDM trainer")
    parser.add_argument("--config", dest="config_path", default="./config/ccdm_geometries.yaml", help="Path to the base YAML config")
    parser.add_argument("--project", dest="project", default=None, help="Weights & Biases project name")
    parser.add_argument("--entity", dest="entity", default=None, help="Weights & Biases entity/user")
    parser.add_argument("--method", dest="method", default="bayes", choices=["bayes", "grid", "random"], help="Sweep search strategy")
    parser.add_argument("--count", dest="count", type=int, default=None, help="Optional number of runs to execute")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Seed passed to the training run")
    parser.add_argument("--sweep-name", dest="sweep_name", default=None, help="Optional sweep display name")
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Could not find config file at '{args.config_path}'")

    base_config = _load_config(args.config_path)
    sweep_config = build_sweep_config(base_config, args.method, args.sweep_name)

    _prepare_wandb_env(args.project, args.entity)

    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)

    print(f"[ccdm_sweep] Created sweep '{sweep_id}'. Starting agent...")

    def _sweep_body() -> None:
        run = wandb.init(project=args.project, entity=args.entity)
        if run is None:
            raise RuntimeError("wandb.init() returned None during sweep execution")
        try:
            updated_config = apply_sweep_config(base_config, dict(run.config))
            run.config.update(updated_config, allow_val_change=True)
            run.config.update({"sweep_seed": args.seed}, allow_val_change=True)
            run_training_with_config(updated_config, args.seed, wandb_run=run)
        finally:
            run.finish()

    wandb.agent(sweep_id, function=_sweep_body, count=args.count, project=args.project, entity=args.entity)


if __name__ == "__main__":
    main()
