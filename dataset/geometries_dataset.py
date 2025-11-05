import ast
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.diffusion_utils import load_latents


class GeometriesDataset(torch.utils.data.Dataset):
    def __init__(self, im_path, im_size, im_channels, use_latents=False,
                 latent_path=None, condition_config=None, max_samples: Optional[int] = None):
        self.im_size = im_size
        self.im_channels = im_channels
        self.max_samples = max_samples

        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False

        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        self._resize_transform = torchvision.transforms.Resize((self.im_size, self.im_size))
        self._train_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        if self.max_samples is not None:
            self.images = self.images[: self.max_samples]
            self.labels = self.labels[: self.max_samples]

        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"images path {im_path} does not exist"
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name))
            for fname in fnames:
                ims.append(fname)
                label_path = fname.replace('images', 'labels').replace('.png', '.txt').replace('image_', 'labels_')
                labels.append(label_path)
        print(f'Found {len(ims)} images')
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cond_inputs = {}
        if 'tensor' in self.condition_types:
            with open(self.labels[index], 'r') as file:
                line = file.readline().strip()
            array = ast.literal_eval(line)
            label_tensor = torch.tensor(array)
            cond_inputs['tensor'] = label_tensor

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            if self.im_channels == 1:
                im = im.convert('L')
            else:
                im = im.convert('RGB')
            im = self._resize_transform(im)
            im_tensor = torchvision.transforms.ToTensor()(im)

            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs

    def _normalize_labels(self, labels: np.ndarray) -> np.ndarray:
        min_vals = labels.min(axis=0, keepdims=True)
        max_vals = labels.max(axis=0, keepdims=True)
        ranges = np.where(max_vals > min_vals, max_vals - min_vals, 1.0)
        normalized = (labels - min_vals) / ranges
        return normalized.astype(np.float32)

    def load_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._train_cache is not None:
            return self._train_cache

        images: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        for image_path, label_path in zip(self.images, self.labels):
            image = Image.open(image_path)
            if self.im_channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            image = self._resize_transform(image)
            image_array = np.array(image, dtype=np.uint8)
            if image_array.ndim == 2:
                image_array = image_array[:, :, None]
            if image_array.shape[-1] != self.im_channels:
                raise ValueError(
                    f"Expected {self.im_channels} channels but found {image_array.shape[-1]} for {image_path}"
                )
            image_array = np.transpose(image_array, (2, 0, 1))
            images.append(image_array)

            with open(label_path, 'r', encoding='utf-8') as file:
                line = file.readline().strip()
            label_values = np.asarray(ast.literal_eval(line), dtype=np.float32)
            label_values = np.atleast_1d(label_values)
            labels.append(label_values)

        train_images = np.stack(images, axis=0)
        raw_labels = np.stack(labels, axis=0).astype(np.float32)
        normalized_labels = self._normalize_labels(raw_labels)

        self._train_cache = (train_images, raw_labels, normalized_labels)
        return self._train_cache


class GeometriesDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            im_size: int = 128,
            im_channels: int = 1,
            batch_size: int = 32,
            num_workers: int = 4,
            use_latents: bool = False,
            latent_path: Optional[str] = None,
            condition_config: Optional[dict] = None,
            train_val_test_split: tuple = (0.7, 0.15, 0.15),
            max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.im_size = im_size
        self.im_channels = im_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_latents = use_latents
        self.latent_path = latent_path
        self.condition_config = condition_config
        self.train_val_test_split = train_val_test_split
        self.max_samples = max_samples

        # Validate split ratio
        assert sum(train_val_test_split) == 1.0, "Split ratios must sum to 1"

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Load the full dataset
        if self.dataset is None:
            self.dataset = GeometriesDataset(
                im_path=self.data_dir,
                im_size=self.im_size,
                im_channels=self.im_channels,
                use_latents=self.use_latents,
                latent_path=self.latent_path,
                condition_config=self.condition_config,
                max_samples=self.max_samples,
            )

        # Calculate lengths for splits
        total_length = len(self.dataset)
        train_length = int(self.train_val_test_split[0] * total_length)
        val_length = int(self.train_val_test_split[1] * total_length)
        test_length = total_length - train_length - val_length

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = \
            random_split(self.dataset, [train_length, val_length, test_length])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
