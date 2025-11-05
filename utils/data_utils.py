"""Utilities for preparing dataset tensors for training."""

from typing import List, Optional, Tuple, Union

import numpy as np

from dataset.geometries_dataset import GeometriesDataModule


def _flatten_condition_tensor(
    cond_inputs: dict, label_index: Optional[int]
) -> Union[np.ndarray, float]:
    if "tensor" not in cond_inputs:
        raise ValueError("Conditional tensor missing from dataset output")

    cond_tensor = cond_inputs["tensor"].float().cpu()
    cond_flat = cond_tensor.reshape(-1)
    if cond_flat.numel() == 0:
        raise ValueError("Condition tensor must contain at least one element")

    if cond_flat.numel() == 1 or label_index is not None:
        selected_index = 0 if label_index is None else int(label_index)
        if selected_index >= cond_flat.shape[0]:
            raise ValueError("label_index is out of bounds for condition tensor")
        return cond_flat[selected_index].item()

    return cond_flat.numpy().astype(np.float32)


def collect_images_and_labels(
    data_module: GeometriesDataModule,
    *,
    label_index: Optional[int] = None,
    normalize_labels: bool = True,
    return_metadata: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict]]:
    """Iterate through the training dataset to obtain NumPy arrays."""

    data_module.setup()

    train_subset = data_module.train_dataset
    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for idx in train_subset.indices:
        sample = train_subset.dataset[idx]
        if not isinstance(sample, tuple):
            raise ValueError("Geometries dataset must return (image, cond) tuples")
        image_tensor, cond_inputs = sample

        label_value = _flatten_condition_tensor(cond_inputs, label_index)
        if isinstance(label_value, float):
            labels.append(np.array(label_value, dtype=np.float32))
        else:
            labels.append(label_value)

        image_np = (image_tensor.numpy() * 255.0).astype(np.float32)
        images.append(image_np)

    if len(images) == 0:
        raise ValueError("No images were collected from the data module")

    images_np = np.stack(images)

    if len(labels) == 0:
        raise ValueError("No labels were collected for training")

    if any(np.asarray(label).ndim > 1 for label in labels):
        raise ValueError("Labels must be scalars or 1D vectors")

    labels_np = (
        np.stack(labels).astype(np.float32)
        if any(np.asarray(label).ndim == 1 for label in labels)
        else np.array(labels, dtype=np.float32)
    )

    label_min = labels_np.min(axis=0)
    label_max = labels_np.max(axis=0)

    if normalize_labels:
        label_span = label_max - label_min
        if np.any(np.isclose(label_span, 0.0)):
            raise ValueError("Labels must span a non-zero range when normalisation is enabled")
        labels_np = (labels_np - label_min) / label_span
        label_min = np.zeros_like(label_min, dtype=np.float32)
        label_max = np.ones_like(label_max, dtype=np.float32)

    metadata = {
        "label_min": np.array(label_min, dtype=np.float32),
        "label_max": np.array(label_max, dtype=np.float32),
        "num_samples": images_np.shape[0],
    }

    if return_metadata:
        return images_np, labels_np, metadata
    return images_np, labels_np

