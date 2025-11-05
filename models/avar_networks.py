"""Utility helpers to access the original AVAR convolutional networks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Tuple


@dataclass(frozen=True)
class _NetworkSpec:
    file_name: str
    generator_cls: str
    discriminator_cls: str


@dataclass(frozen=True)
class _AuxiliarySpec:
    file_name: str
    builder: str


_ROOT = Path(__file__).resolve().parent.parent / "CcGAN-AVAR-main" / "models"

_NETWORK_SPECS = {
    "sngan": _NetworkSpec("sngan.py", "sngan_generator", "sngan_discriminator"),
    "sagan": _NetworkSpec("sagan.py", "sagan_generator", "sagan_discriminator"),
    "biggan": _NetworkSpec("biggan.py", "biggan_generator", "biggan_discriminator"),
    "biggan-deep": _NetworkSpec("biggan_deep.py", "biggan_deep_generator", "biggan_deep_discriminator"),
    "dcgan": _NetworkSpec("dcgan.py", "dcgan_generator", "dcgan_discriminator"),
}

_AUXILIARY_SPECS = {
    "resnet18": _AuxiliarySpec("resnet_aux_regre.py", "resnet18_aux_regre"),
    "resnet34": _AuxiliarySpec("resnet_aux_regre.py", "resnet34_aux_regre"),
    "resnet50": _AuxiliarySpec("resnet_aux_regre.py", "resnet50_aux_regre"),
}


def _load_module(module_name: str, file_name: str):
    path = _ROOT / file_name
    if not path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Unable to locate AVAR model definition at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib failure
        raise ImportError(f"Failed to create a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _get_network_module(file_name: str):
    return _load_module(f"ccgan_avar_{file_name.replace('.', '_')}", file_name)


def load_network_constructors(net_name: str) -> Tuple[Callable[..., object], Callable[..., object]]:
    """Return the generator and discriminator constructors for ``net_name``."""

    key = net_name.lower()
    if key not in _NETWORK_SPECS:
        valid = ", ".join(sorted(_NETWORK_SPECS))
        raise ValueError(f"Unknown AVAR network '{net_name}'. Available choices: {valid}")
    spec = _NETWORK_SPECS[key]
    module = _get_network_module(spec.file_name)
    try:
        generator = getattr(module, spec.generator_cls)
        discriminator = getattr(module, spec.discriminator_cls)
    except AttributeError as exc:  # pragma: no cover - corrupted checkout
        raise ImportError(f"AVAR module {spec.file_name} does not expose required classes") from exc
    return generator, discriminator


@lru_cache(maxsize=None)
def load_auxiliary_builder(arch: str) -> Callable[..., object]:
    """Return a constructor for the requested auxiliary regression backbone."""

    key = arch.lower()
    if key not in _AUXILIARY_SPECS:
        valid = ", ".join(sorted(_AUXILIARY_SPECS))
        raise ValueError(f"Unsupported auxiliary regression architecture '{arch}'. Choices: {valid}")
    spec = _AUXILIARY_SPECS[key]
    module = _get_network_module(spec.file_name)
    try:
        builder = getattr(module, spec.builder)
    except AttributeError as exc:  # pragma: no cover - corrupted checkout
        raise ImportError(f"Failed to locate builder '{spec.builder}' in {spec.file_name}") from exc
    return builder


__all__ = ["load_network_constructors", "load_auxiliary_builder"]

