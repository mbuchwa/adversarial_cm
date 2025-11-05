import importlib
import sys
import types
from pathlib import Path


def load_joint_sweep():
    """Import ``joint_sweep`` with light-weight stubs for heavy deps."""
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    stubs = {
        "torch": types.SimpleNamespace(device=lambda *args, **kwargs: None),
        "wandb": types.SimpleNamespace(sweep=lambda *a, **k: None, agent=lambda *a, **k: None),
        "yaml": types.SimpleNamespace(safe_load=lambda *a, **k: None, safe_dump=lambda *a, **k: None),
        "joint_training": types.SimpleNamespace(),
    }
    restored = {}
    for name, module in stubs.items():
        restored[name] = sys.modules.get(name)
        sys.modules[name] = module
    try:
        module = importlib.import_module("joint_sweep")
        return importlib.reload(module)
    finally:
        for name, original in restored.items():
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original


def test_apply_sweep_config_prefers_explicit_parameter_values():
    joint_sweep = load_joint_sweep()
    base = {
        "train_params": {"ldm_lr": 0.0001},
        "ldm_params": {"time_emb_dim": 256},
    }
    run_config = {
        "train_params": {"ldm_lr": 0.0001},
        "ldm_params": {"time_emb_dim": 256},
        "train.ldm_lr": 0.0008,
        "ldm.time_emb_dim": 384,
    }

    updated = joint_sweep.apply_sweep_config(base, run_config)

    assert updated["train_params"]["ldm_lr"] == 0.0008
    assert updated["ldm_params"]["time_emb_dim"] == 384


def test_apply_sweep_config_supports_sanitised_keys():
    joint_sweep = load_joint_sweep()
    base = {
        "train_params": {"lambda_adv": 0.1},
    }
    run_config = {
        "train_params": {"lambda_adv": 0.1},
        "train_lambda_adv": 0.3,
    }

    updated = joint_sweep.apply_sweep_config(base, run_config)

    assert updated["train_params"]["lambda_adv"] == 0.3


def test_apply_sweep_config_keeps_down_and_up_layers_in_sync():
    joint_sweep = load_joint_sweep()
    base = {
        "ldm_params": {"num_down_layers": 3, "num_up_layers": 3},
    }
    run_config = {
        "ldm.num_down_layers": 4,
        "ldm.num_up_layers": 2,
    }

    updated = joint_sweep.apply_sweep_config(base, run_config)

    assert updated["ldm_params"]["num_down_layers"] == 4
    assert updated["ldm_params"]["num_up_layers"] == 4


def test_apply_sweep_config_updates_channel_dimensions():
    joint_sweep = load_joint_sweep()
    base = {
        "ldm_params": {
            "down_channels": [128, 128, 128, 128],
            "mid_channels": [128, 128],
            "conv_out_channels": 128,
            "num_dimensions": 128,
        }
    }
    run_config = {
        "ldm.num_dimensions": 192,
    }

    updated = joint_sweep.apply_sweep_config(base, run_config)

    assert updated["ldm_params"]["num_dimensions"] == 192
    assert updated["ldm_params"]["down_channels"] == [192, 192, 192, 192]
    assert updated["ldm_params"]["mid_channels"] == [192, 192]
    assert updated["ldm_params"]["conv_out_channels"] == 192
