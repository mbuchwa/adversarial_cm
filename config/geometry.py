"""Shared geometry label schema configuration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

_DEFAULT_CONTINUOUS_BOUNDS: Dict[str, Tuple[float, float]] = {
    "x_pos": (0.2, 0.8),
    "y_pos": (0.2, 0.8),
    "size": (0.1, 0.2),
    "rotation": (0.0, 1.0),
}

_DEFAULT_CONTINUOUS_ORDER: Tuple[str, ...] = tuple(_DEFAULT_CONTINUOUS_BOUNDS.keys())


def _coerce_bounds(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError("Geometry bounds must contain exactly two values: (min, max)")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(
            f"Geometry bound lower value {lower} exceeds upper value {upper}"
        )
    return lower, upper


@dataclass
class GeometryLabelSchema:
    """Configuration describing the geometry label layout and bounds."""

    one_hot_dim: int = 4
    continuous_bounds: Mapping[str, Sequence[float]] = field(
        default_factory=lambda: dict(_DEFAULT_CONTINUOUS_BOUNDS)
    )
    continuous_order: Iterable[str] = _DEFAULT_CONTINUOUS_ORDER

    def __post_init__(self) -> None:
        if self.one_hot_dim < 0:
            raise ValueError("one_hot_dim must be non-negative")

        bounds: MutableMapping[str, Tuple[float, float]]
        if isinstance(self.continuous_bounds, dict):
            bounds = dict(self.continuous_bounds)
        else:
            bounds = {key: tuple(value) for key, value in self.continuous_bounds.items()}

        coerced_bounds = {
            key: _coerce_bounds(value)
            for key, value in bounds.items()
        }

        order_tuple = tuple(self.continuous_order) if self.continuous_order else tuple(
            coerced_bounds.keys()
        )
        missing = [key for key in order_tuple if key not in coerced_bounds]
        if missing:
            raise KeyError(
                "Continuous bounds missing required keys: " + ", ".join(missing)
            )
        extra = [key for key in coerced_bounds if key not in order_tuple]
        if extra:
            order_tuple = order_tuple + tuple(extra)

        self.continuous_bounds = coerced_bounds
        self.continuous_order = order_tuple

    def lower_bounds(self) -> Tuple[float, ...]:
        return tuple(
            float(self.continuous_bounds[name][0]) for name in self.continuous_order
        )

    def upper_bounds(self) -> Tuple[float, ...]:
        return tuple(
            float(self.continuous_bounds[name][1]) for name in self.continuous_order
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "one_hot_dim": int(self.one_hot_dim),
            "continuous_bounds": {
                key: (float(bounds[0]), float(bounds[1]))
                for key, bounds in self.continuous_bounds.items()
            },
            "continuous_order": tuple(self.continuous_order),
        }
