import math
from typing import Dict, Any

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from PIL import Image

from dataset.geometries_dataset import GeometriesDataset

from CCDM_unified.label_embedding import LabelEmbed
from ccdm_training import _instantiate_module
from models.pl_ccdm import LightningCCDM


def _legacy_scalar_sinusoidal(labels: torch.Tensor, embed_dim: int) -> torch.Tensor:
    max_period = 10000
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=labels.device)
        / half
    )
    args = labels[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    embedding = (embedding + 1) / 2
    return embedding


def test_scalar_sinusoidal_embedding_matches_legacy(tmp_path):
    embed_dim = 8
    labels = torch.linspace(0, 1, steps=5)
    embedder = LabelEmbed(
        dataset=None,
        path_y2h=str(tmp_path / "y2h"),
        path_y2cov=str(tmp_path / "y2cov"),
        y2h_type="sinusoidal",
        y2cov_type="gaussian",
        h_dim=embed_dim,
        cov_dim=16,
        device="cpu",
    )

    new_embedding = embedder.fn_y2h(labels)
    expected = _legacy_scalar_sinusoidal(labels, embed_dim)
    assert torch.allclose(new_embedding, expected)


def test_vector_embeddings_return_expected_shapes(tmp_path):
    batch_size = 4
    cond_dim = 3
    embed_dim = 12
    cov_dim = 24
    labels = torch.rand(batch_size, cond_dim)

    embedder = LabelEmbed(
        dataset=None,
        path_y2h=str(tmp_path / "y2h_vec"),
        path_y2cov=str(tmp_path / "y2cov_vec"),
        y2h_type="sinusoidal",
        y2cov_type="gaussian",
        h_dim=embed_dim,
        cov_dim=cov_dim,
        device="cpu",
    )

    emb_h = embedder.fn_y2h(labels)
    emb_cov = embedder.fn_y2cov(labels)

    assert emb_h.shape == (batch_size, embed_dim)
    assert emb_cov.shape == (batch_size, cov_dim)

    gaussian_embedder = LabelEmbed(
        dataset=None,
        path_y2h=str(tmp_path / "y2h_gauss"),
        path_y2cov=str(tmp_path / "y2cov_gauss"),
        y2h_type="gaussian",
        y2cov_type="gaussian",
        h_dim=embed_dim,
        cov_dim=cov_dim,
        device="cpu",
    )

    emb_gauss = gaussian_embedder.fn_y2h(labels)
    assert emb_gauss.shape == (batch_size, embed_dim)


def test_resnet_label_embedding_requires_dataset(tmp_path):
    label_embed_kwargs = {
        "y2h_type": "resnet",
        "path_y2h": str(tmp_path / "resnet_y2h"),
        "path_y2cov": str(tmp_path / "resnet_y2cov"),
    }

    with pytest.raises(
        ValueError,
        match=r"dataset implementing load_train_data\(\)",
    ):
        LightningCCDM(
            image_size=32,
            in_channels=3,
            results_folder=str(tmp_path / "results"),
            label_embed_kwargs=label_embed_kwargs,
        )


def test_geometries_resnet_instantiation_loads_training_data(tmp_path, monkeypatch):
    data_root = tmp_path / "geometries"
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for idx in range(3):
        image = Image.fromarray(np.full((16, 16), fill_value=idx * 16, dtype=np.uint8), mode="L")
        image_path = images_dir / f"image_{idx}.png"
        image.save(image_path)

        label_path = labels_dir / f"labels_{idx}.txt"
        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.write(str([float(idx), float(idx + 1)]))

    captured: Dict[str, Any] = {}

    class DummyLabelEmbed:
        def __init__(self, *, dataset=None, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            if dataset is not None:
                captured["train_data"] = dataset.load_train_data()
            self.fn_y2h = lambda labels: labels
            self.fn_y2cov = lambda labels: labels

    monkeypatch.setattr("models.pl_ccdm.LabelEmbed", DummyLabelEmbed)

    config = {
        "dataset_params": {
            "im_path": str(images_dir),
            "im_channels": 1,
            "im_size": 16,
            "condition_config": {"condition_types": ["tensor"]},
        },
        "ccdm_params": {
            "image_size": 16,
            "in_channels": 1,
            "train_lr": 1e-4,
            "adam_betas": [0.9, 0.99],
            "results_folder": str(tmp_path / "results"),
        },
        "label_embedding_params": {
            "y2h_embed_type": "resnet",
            "y2cov_embed_type": "resnet",
            "path_y2h": str(tmp_path / "y2h"),
            "path_y2cov": str(tmp_path / "y2cov"),
        },
        "unet_params": {
            "in_channels": 1,
            "image_size": 16,
        },
        "diffusion_params": {},
    }

    module = _instantiate_module(config)

    assert isinstance(module, LightningCCDM)
    assert "train_data" in captured
    train_images, raw_labels, norm_labels = captured["train_data"]
    assert train_images.shape == (3, 1, 16, 16)
    assert raw_labels.shape == (3, 2)
    assert np.all(norm_labels >= 0.0)
    assert np.all(norm_labels <= 1.0)


def test_resnet_label_embed_handles_multidimensional_labels(tmp_path, monkeypatch):
    data_root = tmp_path / "geom_multi"
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for idx in range(4):
        image = Image.fromarray(np.full((8, 8), fill_value=idx * 32, dtype=np.uint8), mode="L")
        image_path = images_dir / f"image_{idx}.png"
        image.save(image_path)

        label_path = labels_dir / f"labels_{idx}.txt"
        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.write(str([float(idx), float(idx + 1)]))

    dataset = GeometriesDataset(
        im_path=str(images_dir),
        im_size=8,
        im_channels=1,
        condition_config={"condition_types": ["tensor"]},
    )

    captured_label_shapes = []
    captured_unique_shapes = []

    def fake_train_resnet(*, net, net_name, trainloader, epochs, resume_epoch, lr_base, lr_decay_factor, lr_decay_epochs, weight_decay, path_to_ckpt, device, label_shape):
        batch_images, batch_labels = next(iter(trainloader))
        captured_label_shapes.append(tuple(batch_labels.shape))
        outputs, _ = net(batch_images.float())
        assert outputs.shape[0] == batch_labels.shape[0]
        assert tuple(outputs.shape[1:]) == tuple(label_shape)
        return net

    def fake_train_mlp(unique_labels_norm, model_mlp, model_name, model_h2y, epochs, lr_base, lr_decay_factor, lr_decay_epochs, weight_decay, batch_size, device, label_shape):
        captured_unique_shapes.append(unique_labels_norm.shape)
        flattened = torch.from_numpy(unique_labels_norm.reshape(unique_labels_norm.shape[0], -1)).float()
        hidden = model_mlp(flattened)
        recon = model_h2y(hidden)
        assert recon.shape[1:] == tuple(label_shape)
        return model_mlp

    monkeypatch.setattr("CCDM_unified.label_embedding.train_resnet", fake_train_resnet)
    monkeypatch.setattr("CCDM_unified.label_embedding.train_mlp", fake_train_mlp)

    embedder = LabelEmbed(
        dataset=dataset,
        path_y2h=str(tmp_path / "multi_y2h"),
        path_y2cov=str(tmp_path / "multi_y2cov"),
        y2h_type="resnet",
        y2cov_type="resnet",
        h_dim=16,
        cov_dim=12,
        batch_size=2,
        nc=1,
        device="cpu",
    )

    assert captured_label_shapes
    assert captured_unique_shapes
    assert captured_label_shapes[0][1:] == (2,)
    assert captured_unique_shapes[0][1:] == (2,)

    probe_labels = torch.rand(3, 2)
    h_emb = embedder.fn_y2h(probe_labels)
    cov_emb = embedder.fn_y2cov(probe_labels)

    assert h_emb.shape == (3, 16)
    assert cov_emb.shape == (3, 12)
