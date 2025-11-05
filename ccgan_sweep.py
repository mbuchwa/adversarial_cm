"""Utility to launch Weights & Biases sweeps for the CcGAN baseline."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml

import ccgan_training

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

import wandb


PARAMETER_PATHS: Mapping[str, Sequence[str]] = {
    "model.latent_dim": ("model", "latent_dim"),
    "model.hidden_dim": ("model", "hidden_dim"),
    "model.generator_layers": ("model", "generator_layers"),
    "model.discriminator_layers": ("model", "discriminator_layers"),
    "model.radius": ("model", "radius"),
    "optimisation.generator_lr": ("optimisation", "generator_lr"),
    "optimisation.discriminator_lr": ("optimisation", "discriminator_lr"),
    "optimisation.batch_size_disc": ("optimisation", "batch_size_disc"),
    "optimisation.batch_size_gene": ("optimisation", "batch_size_gene"),
    "training.max_steps": ("training", "max_steps"),
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
    model_cfg = config.get("model", {})
    optimisation_cfg = config.get("optimisation", {})
    training_cfg = config.get("training", {})

    latent_dim = int(model_cfg.get("latent_dim", 128))
    hidden_dim = int(model_cfg.get("hidden_dim", 1024))
    generator_layers = int(model_cfg.get("generator_layers", 8))
    discriminator_layers = int(model_cfg.get("discriminator_layers", 6))
    radius = float(model_cfg.get("radius", 1.0))
    generator_lr = float(optimisation_cfg.get("generator_lr", 2e-4))
    discriminator_lr = float(optimisation_cfg.get("discriminator_lr", 2e-4))
    batch_size_disc = int(optimisation_cfg.get("batch_size_disc", 128))
    batch_size_gene = int(optimisation_cfg.get("batch_size_gene", 128))
    max_steps = int(training_cfg.get("max_steps", 100_000))

    parameter_values: Dict[str, Any] = {
        "model.latent_dim": {"values": sorted({64, latent_dim, 256})},
        "model.hidden_dim": {"values": sorted({512, hidden_dim, 2048})},
        "model.generator_layers": {"values": sorted({6, generator_layers, 10})},
        "model.discriminator_layers": {"values": sorted({4, discriminator_layers, 8})},
        "model.radius": {"values": sorted({round(v, 2) for v in (0.9, radius, 1.1) if v > 0})},
        "optimisation.generator_lr": {"values": _safe_factor_values(generator_lr, (0.5, 1.0, 1.5))},
        "optimisation.discriminator_lr": {"values": _safe_factor_values(discriminator_lr, (0.5, 1.0, 1.5))},
        "optimisation.batch_size_disc": {"values": sorted({64, batch_size_disc, 128, 96})},
        "optimisation.batch_size_gene": {"values": sorted({64, batch_size_gene, 128, 96})},
        "training.max_steps": {"values": sorted({50_000, max_steps, 150_000})},
    }

    for key, value_dict in parameter_values.items():
        values = value_dict["values"]
        if not values:
            raise ValueError(f"Parameter '{key}' must define at least one value for the sweep.")
    return parameter_values


def build_sweep_config(config: Mapping[str, Any], method: str, sweep_name: str | None = None) -> Dict[str, Any]:
    parameter_values = _build_parameter_values(config)
    sweep_config: Dict[str, Any] = {
        "name": sweep_name or "ccgan_sweep",
        "method": method,
        "metric": {"name": "train/g_loss", "goal": "minimize"},
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
        ccgan_training.train(args, wandb_run=wandb_run)
    finally:
        os.remove(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Weights & Biases sweep for the CcGAN baseline")
    parser.add_argument("--config", dest="config_path", default="./config/ccgan_geometries.yaml", help="Path to the base YAML config")
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

    print(f"[ccgan_sweep] Created sweep '{sweep_id}'. Starting agent...")

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
