from typing import Optional

import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataset.geometry_generation import RandomImageGenerator
from imquality import brisque
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor

from config.geometry import GeometryLabelSchema


class ImageSimilarityMetrics:
    def __init__(self, device='cpu', geometry_label_schema: Optional[GeometryLabelSchema] = None):
        self.device = device
        self.geometry_label_schema = geometry_label_schema or GeometryLabelSchema()
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)

    def calculate_batch_mse(self, real_images, generated_images):
        """
        Calculate the Mean Squared Error (MSE) loss between generated images and real images for a batch.

        Args:
            real_images (torch.Tensor): Batch of real images (shape: [batch_size, C, H, W])
            generated_images (torch.Tensor): Batch of generated images (shape: [batch_size, C, H, W])

        Returns:
            tuple: Mean MSE loss for the batch, and per-image MSEs
        """
        # Ensure both real and generated images are of the same shape
        assert real_images.shape == generated_images.shape, "Shape mismatch between real and generated images"

        # Calculate MSE for each image in the batch
        mse_per_image = torch.mean((real_images - generated_images) ** 2, dim=(1, 2, 3))  # [B]

        # Calculate the overall (mean) MSE for the batch
        mean_mse = torch.mean(mse_per_image)  # scalar value

        return mean_mse.item(), mse_per_image.tolist()

    def calculate_ssim(self, real_images, generated_images):
        """
        Calculate SSIM between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            float: Mean SSIM across batch
            torch.Tensor: SSIM per image
        """
        batch_size = real_images.shape[0]
        ssim_scores = []

        # Convert to numpy and ensure proper range [0, 1]
        real_np = real_images.cpu().numpy()
        gen_np = generated_images.cpu().numpy()

        for i in range(batch_size):
            score = ssim(real_np[i, 0], gen_np[i, 0],
                         data_range=1.0,
                         gaussian_weights=True,
                         sigma=1.5,
                         use_sample_covariance=False)
            ssim_scores.append(score)

        ssim_scores = torch.tensor(ssim_scores, device=self.device)
        return torch.mean(ssim_scores).item(), ssim_scores

    def calculate_psnr(self, real_images, generated_images):
        """
        Calculate PSNR between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            float: Mean PSNR across batch
            torch.Tensor: PSNR per image
        """
        batch_size = real_images.shape[0]
        psnr_scores = []

        # Convert to numpy and ensure proper range [0, 1]
        real_np = real_images.cpu().numpy()
        gen_np = generated_images.cpu().numpy()

        for i in range(batch_size):
            score = psnr(real_np[i, 0], gen_np[i, 0], data_range=1.0)
            psnr_scores.append(score)

        psnr_scores = torch.tensor(psnr_scores, device=self.device)
        return torch.mean(psnr_scores).item(), psnr_scores

    def calculate_lpips(self, real_images, generated_images):
        """
        Calculate LPIPS distance between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            float: Mean LPIPS across batch
            torch.Tensor: LPIPS per image
        """
        with torch.no_grad():
            real_images_lpips = real_images.to(self.device)
            generated_images_lpips = generated_images.to(self.device)

            # Repeat grayscale channel to match LPIPS input requirements
            if real_images_lpips.shape[1] == 1:
                real_images_lpips = real_images_lpips.repeat(1, 3, 1, 1)
                generated_images_lpips = generated_images_lpips.repeat(1, 3, 1, 1)

            distances = self.lpips_model(real_images_lpips, generated_images_lpips)
            distances = distances.squeeze().detach().cpu()

        return torch.mean(distances).item(), distances

    def calculate_all_metrics(self, image_conditions, generated_images):
        """
        Calculate all similarity metrics between real and generated images.

        Args:
            real_images (torch.Tensor): Shape [B, C, H, W]
            generated_images (torch.Tensor): Shape [B, C, H, W]

        Returns:
            dict: Dictionary containing mean and per-image metrics
        """

        generated_images = generated_images.cpu()
        batch_size = len(generated_images)
        generator = RandomImageGenerator(
            image_shape=generated_images.shape[-2:],
            geometry_label_schema=self.geometry_label_schema,
        )

        # List to store real images generated from conditions
        real_images_list = []

        for i in range(batch_size):
            # Extract conditions for the current image
            condition = image_conditions[i]

            # Create a real image using the RandomImageGenerator
            real_image, _ = generator.generate_image_from_condition(condition.cpu())

            # Convert real_image (assumed to be a NumPy array) to a PyTorch tensor and add a channel dimension
            real_image_tensor = torch.tensor(real_image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

            # Append to list of real images
            real_images_list.append(real_image_tensor)

        # Concatenate all real images into a single tensor [B, 1, H, W]
        real_images = torch.cat(real_images_list, dim=0).unsqueeze(1)
        mean_mse, mse_per_image = self.calculate_batch_mse(real_images, generated_images)
        mean_ssim, ssim_per_image = self.calculate_ssim(real_images, generated_images)
        mean_psnr, psnr_per_image = self.calculate_psnr(real_images, generated_images)
        mean_lpips, lpips_per_image = self.calculate_lpips(real_images, generated_images)

        return real_images, {
            'mse': {'mean': mean_mse, 'per_image': mse_per_image},
            'ssim': {'mean': mean_ssim, 'per_image': ssim_per_image},
            'psnr': {'mean': mean_psnr, 'per_image': psnr_per_image},
            'lpips': {'mean': mean_lpips, 'per_image': lpips_per_image}
        }


class ImageEvaluationMetrics:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device).eval()

    def compute_metrics(self, generated_images: torch.Tensor, reference_images: torch.Tensor = None):
        """
        Args:
            generated_images (Tensor): shape (B, C, H, W), range [0, 1]
            reference_images (Tensor): optional, shape (B, C, H, W), range [0, 1]
        Returns:
            dict: {"sfid": float, "niqe": float, "diversity": float}
        """
        results = {}

        # --- SFID ---
        if reference_images is not None:
            sfid = self._compute_sfid(generated_images, reference_images)
            results['sfid'] = sfid

        # # --- NIQE ---
        # niqe_score = self._compute_niqe(generated_images)
        # results['niqe'] = niqe_score

        # --- Diversity ---
        diversity_score = self._compute_diversity(generated_images)
        results['diversity'] = diversity_score

        return results

    def _compute_sfid(self, generated, reference):
        """
        Computes the mean LPIPS between generated and reference images.
        """
        assert generated.shape == reference.shape, "Image shapes must match"
        dists = []
        for i in range(generated.size(0)):
            dist = self.lpips_fn(generated[i:i+1], reference[i:i+1])
            dists.append(dist.item())
        return float(np.mean(dists))

    # def _compute_niqe(self, images):
    #     """
    #     Computes NIQE score (lower is better).
    #     """
    #     to_pil = ToPILImage()
    #     rescaler = Resize((256, 256))  # Standard BRISQUE input size
    #
    #     scores = []
    #     for img in images.cpu().clamp(0, 1):
    #         img_pil = to_pil(img)  # Convert to PIL
    #         img_resized = rescaler(img_pil)  # Resize to 256x256
    #
    #         # Convert to numpy and score
    #         score = brisque.score(img_resized)
    #         scores.append(score)
    #
    #     return float(np.mean(scores))

    def _compute_diversity(self, images):
        """
        Computes average LPIPS distance between pairs of images (higher is better).
        """
        B = images.size(0)
        dists = []
        for i in range(B):
            for j in range(i + 1, B):
                dist = self.lpips_fn(images[i:i+1], images[j:j+1])
                dists.append(dist.item())
        return float(np.mean(dists)) if dists else 0.0
