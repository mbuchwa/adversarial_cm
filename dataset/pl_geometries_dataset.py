import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
import torch
import ast
from typing import Optional


class GeometriesDataset(torch.utils.data.Dataset):
    def __init__(self, im_path, im_size, im_channels, condition_config=None):
        self.im_size = im_size
        self.im_channels = im_channels

        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False

        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"images path {im_path} does not exist"
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpeg')))
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
            im = im.convert('L')  # grayscale
            resize_transform = torchvision.transforms.Resize((128, 128))
            im = resize_transform(im)
            im_tensor = torchvision.transforms.ToTensor()(im)

            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs


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
            train_val_test_split: tuple = (0.7, 0.15, 0.15)
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
                condition_config=self.condition_config
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
