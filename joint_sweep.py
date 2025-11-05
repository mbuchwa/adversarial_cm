"""Utility to launch Weights & Biases sweeps for the joint training pipeline."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence, Tuple
import sys
import torch
import yaml

import joint_training

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

import wandb


PARAMETER_PATHS: Mapping[str, Sequence[str]] = {
    "train.ldm_lr": ("train_params", "ldm_lr"),
    "train.disc_lr": ("train_params", "disc_lr"),
    "train.consistency_lr": ("train_params", "consistency_lr"),
    "train.discriminator_weight": ("train_params", "discriminator_weight"),
    "train.perceptual_weight": ("train_params", "perceptual_weight"),
    "train.discriminator_steps": ("train_params", "discriminator_steps"),
    "train.adv_ramp_up_epochs": ("train_params", "adv_ramp_up_epochs"),
    "train.lambda_adv": ("train_params", "lambda_adv"),
    "train.lambda_adv_min": ("train_params", "lambda_adv_min"),
    "train.lambda_adv_max": ("train_params", "lambda_adv_max"),
    "train.freeze_cm_epoch": ("train_params", "freeze_cm_epoch"),
    "train.rollout_warmup_epochs": ("train_params", "rollout_warmup_epochs"),
    "train.rollout_plateau_patience": ("train_params", "rollout_plateau_patience"),
    "train.rollout_plateau_delta": ("train_params", "rollout_plateau_delta"),
    "ldm.dropout_rate": ("ldm_params", "dropout_rate"),
    "ldm.cond_factor": ("ldm_params", "cond_factor"),
    "ldm.norm_channels": ("ldm_params", "norm_channels"),
    "ldm.time_emb_dim": ("ldm_params", "time_emb_dim"),
    "ldm.num_down_layers": ("ldm_params", "num_down_layers"),
    "ldm.num_mid_layers": ("ldm_params", "num_mid_layers"),
    "ldm.num_up_layers": ("ldm_params", "num_up_layers"),
    "ldm.num_dimensions": ("ldm_params", "num_dimensions"),
    "consistency.weight_decay": ("consistency_params", "weight_decay"),
    "consistency.ema_decay": ("consistency_params", "ema_decay"),
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
    train_cfg = config.get("train_params", {})
    ldm_cfg = config.get("ldm_params", {})
    consistency_cfg = config.get("consistency_params", {})

    lambda_adv_base = float(train_cfg.get("lambda_adv", 0.1))
    lambda_adv_min_base = float(train_cfg.get("lambda_adv_min", 0.01))
    lambda_adv_max_base = float(train_cfg.get("lambda_adv_max", 2.0))
    lambda_adv_values = _safe_factor_values(lambda_adv_base, (0.5, 1.0, 2.0))
    lambda_adv_min_values = [
        value
        for value in _safe_factor_values(lambda_adv_min_base, (0.5, 1.0, 2.0))
        if value <= lambda_adv_max_base
    ]
    if not lambda_adv_min_values:
        lambda_adv_min_values = [round(lambda_adv_min_base, 8)]
    lambda_adv_max_values = _safe_factor_values(lambda_adv_max_base, (0.5, 1.0, 1.5))

    freeze_cm_epoch_base = int(train_cfg.get("freeze_cm_epoch", 50))
    freeze_cm_epoch_values = sorted(
        {max(1, int(round(freeze_cm_epoch_base * factor))) for factor in (0.0, 0.5, 1.0, 1.5, 2.0)}
    )

    rollout_warmup_base = int(train_cfg.get("rollout_warmup_epochs", 30))
    rollout_warmup_values = sorted(
        {max(1, int(round(rollout_warmup_base * factor))) for factor in (0.5, 1.0, 1.5)}
    )

    rollout_plateau_patience_base = int(train_cfg.get("rollout_plateau_patience", 10))
    rollout_plateau_patience_values = sorted(
        {
            max(1, int(round(rollout_plateau_patience_base * factor)))
            for factor in (0.5, 1.0, 1.5)
        }
    )

    rollout_plateau_delta_base = float(train_cfg.get("rollout_plateau_delta", 0.01))
    rollout_plateau_delta_values = _safe_factor_values(rollout_plateau_delta_base, (0.5, 1.0, 2.0))

    num_down_layers_base = int(
        ldm_cfg.get("num_down_layers", max(1, len(ldm_cfg.get("down_channels", [])) - 1))
    )
    num_mid_layers_base = int(
        ldm_cfg.get("num_mid_layers", max(1, len(ldm_cfg.get("mid_channels", []))))
    )
    num_up_layers_base = int(ldm_cfg.get("num_up_layers", num_down_layers_base))

    down_channels = ldm_cfg.get("down_channels", [])
    base_channel_value = None
    if isinstance(down_channels, Sequence):
        for channel in down_channels:
            if channel:
                base_channel_value = int(channel)
                break
    num_dimensions_base = int(ldm_cfg.get("num_dimensions", base_channel_value or 128))

    num_down_layers_values = sorted({max(1, num_down_layers_base + delta) for delta in (-1, 0, 1)})
    if not num_down_layers_values:
        num_down_layers_values = [num_down_layers_base]

    num_up_layers_values = sorted(set(num_down_layers_values) | {max(1, num_up_layers_base)})
    num_mid_layers_values = sorted({max(1, num_mid_layers_base + delta) for delta in (-1, 0, 1)})
    if not num_mid_layers_values:
        num_mid_layers_values = [num_mid_layers_base]

    num_dimensions_values = sorted(
        {
            max(8, int(round(num_dimensions_base * factor)))
            for factor in (0.5, 1.0, 1.5, 2.0)
        }
    )

    parameter_values: Dict[str, Any] = {
        "train.disc_lr": {"values": _safe_factor_values(train_cfg.get("disc_lr", 1e-3), (0.5, 1.5, 2.0))},
        "train.consistency_lr": {"values": _safe_factor_values(train_cfg.get("consistency_lr", 3e-4), (0.5, 1.0, 1.5))},
        "train.perceptual_weight": {"values": sorted({round(train_cfg.get("perceptual_weight", 1.0) * factor, 8) for factor in (0.5, 1.0, 1.5)})},
        "train.discriminator_steps": {"values": sorted({max(1, int(round(train_cfg.get("discriminator_steps", 10) * factor))) for factor in (0.5, 1.0, 1.5)})},
        "train.adv_ramp_up_epochs": {"values": sorted({max(1, int(round(train_cfg.get("adv_ramp_up_epochs", 10) * factor))) for factor in (0.5, 1.0, 2.0)})},
        "train.lambda_adv": {"values": lambda_adv_values},
        "train.lambda_adv_min": {"values": lambda_adv_min_values},
        "train.lambda_adv_max": {"values": lambda_adv_max_values},
        "ldm.dropout_rate": {"values": [0.0, 0.05, 0.1]},
        "ldm.time_emb_dim": {"values": sorted({max(128, int(ldm_cfg.get("time_emb_dim", 256) + delta)) for delta in (0, 128, 256)})},
        "ldm.num_down_layers": {"values": num_down_layers_values},
        "ldm.num_up_layers": {"values": num_up_layers_values},
        "ldm.num_mid_layers": {"values": num_mid_layers_values},
        "ldm.num_dimensions": {"values": num_dimensions_values},
        "consistency.weight_decay": {"values": _safe_factor_values(consistency_cfg.get("weight_decay", 1e-4), (0.5, 1.0, 5.0))},
        "consistency.ema_decay": {"values": sorted({round(val, 5) for val in (0.95, float(consistency_cfg.get("ema_decay", 0.97)), 0.995)})},
        "train.freeze_cm_epoch": {"values": freeze_cm_epoch_values},
        "train.rollout_warmup_epochs": {"values": rollout_warmup_values},
        "train.rollout_plateau_patience": {"values": rollout_plateau_patience_values},
        "train.rollout_plateau_delta": {"values": rollout_plateau_delta_values},
    }

    for key, value_dict in parameter_values.items():
        values = value_dict["values"]
        if not values:
            raise ValueError(f"Parameter '{key}' must define at least one value for the sweep.")
    return parameter_values


def build_sweep_config(config: Mapping[str, Any], method: str, sweep_name: str | None = None,
                       program: str | None = None,) -> Dict[str, Any]:
    parameter_values = _build_parameter_values(config)
    sweep_config: Dict[str, Any] = {
        "name": sweep_name or "joint_model_sweep",
        "method": method,
        "metric": {"name": "val/consistency_loss", "goal": "minimize"},
        "parameters": parameter_values,
    }
    if program:
        sweep_config["command"] = [sys.executable, "-m", program]
    return sweep_config


def _set_nested(config: MutableMapping[str, Any], path: Sequence[str], value: Any) -> None:
    cursor: MutableMapping[str, Any] = config
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]  # type: ignore[assignment]
    cursor[path[-1]] = value


def _candidate_config_keys(path_key: str, path: Sequence[str]) -> Tuple[str, ...]:
    """Return possible config keys that may store the sweep value.

    W&B can expose sweep parameters in multiple formats depending on the API
    that is used to access them. Besides the explicit parameter name defined in
    the sweep configuration (e.g. ``ldm.time_emb_dim``), the config object may
    also expose nested keys that mirror the YAML structure (e.g.
    ``ldm_params.time_emb_dim``) or a sanitised variant where dots are replaced
    by underscores (e.g. ``ldm_time_emb_dim``). This helper returns all
    possible representations so that we can reliably locate the sweep value.
    """

    derived_key = ".".join(path)
    underscored_key = path_key.replace(".", "_")
    keys = [path_key]
    if underscored_key not in keys:
        keys.append(underscored_key)
    if derived_key not in keys:
        keys.append(derived_key)
    return tuple(keys)


def apply_sweep_config(base_config: Mapping[str, Any], sweep_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply sweep parameters to base configuration.

    This function handles the various ways wandb may expose sweep parameters:
    - Original format: "train.disc_lr"
    - Underscored: "train_disc_lr"
    - Nested: sweep_config["train_params"]["disc_lr"]
    """

    def _flatten(
            mapping: Mapping[str, Any], parent: Tuple[str, ...] = ()
    ) -> Iterator[Tuple[str, Any]]:
        for key, value in mapping.items():
            if key.startswith("_"):
                continue
            if isinstance(value, Mapping):
                yield from _flatten(value, parent + (key,))
            else:
                full_key = ".".join(parent + (key,)) if parent else key
                yield full_key, value

    flattened = dict(_flatten(sweep_config))
    updated_config = copy.deepcopy(base_config)

    # Track which parameters were successfully applied
    applied_params = set()

    for key, path in PARAMETER_PATHS.items():
        value_set = False
        actual_value = None

        # Try all candidate keys
        for candidate in _candidate_config_keys(key, path):
            if candidate in flattened:
                actual_value = flattened[candidate]
                value_set = True
                break

        # If not found in flattened, try navigating the nested structure directly
        if not value_set:
            # Try the original dotted key
            if key in sweep_config:
                actual_value = sweep_config[key]
                value_set = True
            # Try navigating nested path (e.g., sweep_config["train_params"]["disc_lr"])
            elif len(path) >= 2:
                cursor = sweep_config
                try:
                    for p in path[:-1]:
                        cursor = cursor[p]
                    if path[-1] in cursor:
                        actual_value = cursor[path[-1]]
                        value_set = True
                except (KeyError, TypeError):
                    pass

        if value_set and actual_value is not None:
            _set_nested(updated_config, path, actual_value)
            applied_params.add(key)
            print(f"[joint_sweep] Applied {key} = {actual_value}")

    # Handle special LDM parameters that need to be synchronized
    ldm_updated = updated_config.get("ldm_params")
    if isinstance(ldm_updated, MutableMapping):
        # Synchronize num_up_layers with num_down_layers
        down_layers = ldm_updated.get("num_down_layers")
        if down_layers is not None:
            ldm_updated["num_up_layers"] = down_layers
            print(f"[joint_sweep] Synchronized num_up_layers = {down_layers}")

        # Update all channel lists with num_dimensions
        dimension_value = ldm_updated.get("num_dimensions")
        try:
            dimension_int = int(dimension_value) if dimension_value is not None else None
        except (TypeError, ValueError):
            dimension_int = None

        if dimension_int is not None and dimension_int > 0:
            def _update_channels(key: str) -> None:
                channels = ldm_updated.get(key)
                if isinstance(channels, list) and channels:
                    ldm_updated[key] = [dimension_int for _ in channels]
                    print(f"[joint_sweep] Updated {key} to [{dimension_int}] * {len(channels)}")

            _update_channels("down_channels")
            _update_channels("mid_channels")
            if "conv_out_channels" in ldm_updated:
                ldm_updated["conv_out_channels"] = dimension_int
                print(f"[joint_sweep] Updated conv_out_channels = {dimension_int}")

    if not applied_params:
        print("[joint_sweep] WARNING: No sweep parameters were applied!")
    else:
        print(f"[joint_sweep] Successfully applied {len(applied_params)} parameters")

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
        joint_training.use_pretrained_consistency = False
        joint_training.use_pretrained_discriminator = False
        joint_training.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        args = argparse.Namespace(config_path=tmp_path, seed=seed)
        joint_training.train(args, wandb_run=wandb_run)
    finally:
        os.remove(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Weights & Biases sweep for the joint model")
    parser.add_argument("--config", dest="config_path", default="./config/geometries_cond.yaml", help="Path to the base YAML config")
    parser.add_argument("--project", dest="project", default='joint_cm_sweep', help="Weights & Biases project name")
    parser.add_argument("--entity", dest="entity", default='marcusbuchwald', help="Weights & Biases entity/user")
    parser.add_argument("--method", dest="method", default="bayes", choices=["bayes", "grid", "random"], help="Sweep search strategy")
    parser.add_argument("--count", dest="count", type=int, default=None, help="Optional number of runs to execute")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Seed passed to the joint training run")
    parser.add_argument("--sweep-name", dest="sweep_name", default=None, help="Optional sweep display name")
    parser.add_argument(
        "--program",
        dest="program",
        default="joint_training",
        choices=["joint_training", "pl_joint_5"],
        help="Entry point module controlled by the sweep (default: joint_training)",
    )
    args = parser.parse_args()

    if args.program != "joint_training":
        print("[joint_sweep] Using joint_training entry point to launch the JointDiffusionTrainer.")

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Could not find config file at '{args.config_path}'")

    base_config = _load_config(args.config_path)
    sweep_config = build_sweep_config(base_config, args.method, args.sweep_name, args.program)

    _prepare_wandb_env(args.project, args.entity)

    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)

    print(f"[joint_sweep] Created sweep '{sweep_id}'. Starting agent...")

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
