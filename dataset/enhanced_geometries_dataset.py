import os
import random
from typing import Optional
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter
from config.geometry import GeometryLabelSchema


class EnhancedGeometryGenerator:
    """
    Progressive realism generator for bridging binary shapes to real images.

    Modes:
    - 'binary': Original binary shapes (phase 1)
    - 'textured': Anti-aliased shapes with textures (phase 2)
    - 'realistic': Shadows, gradients, noise (phase 3)
    """

    def __init__(
            self,
            image_shape=(256, 256),
            max_objects=5,
            fixed_num_objects=True,
            labels_as_text=False,
            use_lighting=False,
            geometry_label_schema: Optional[GeometryLabelSchema] = None,
            realism_mode='textured',  # 'binary', 'textured', 'realistic'
            background_noise_level=0.0,  # 0.0 to 0.3
            antialiasing=True,
    ):
        self.image_shape = image_shape
        self.max_objects = max_objects
        self.fixed_num_objects = fixed_num_objects
        self.shapes = ['circle', 'square', 'triangle', 'star']
        self.realism_mode = realism_mode
        self.background_noise_level = background_noise_level
        self.antialiasing = antialiasing

        self.geometry_one_hot = {shape: np.eye(len(self.shapes))[i] for i, shape in enumerate(self.shapes)}
        self.geometry_label_schema = geometry_label_schema or GeometryLabelSchema()
        self._continuous_bounds = self.geometry_label_schema.continuous_bounds

        self.image_dir = f'./Geometries_{max_objects}_2D_{realism_mode}/images'
        self.label_dir = f'./Geometries_{max_objects}_2D_{realism_mode}/labels'
        self.labels_as_text = labels_as_text
        self.use_lighting = use_lighting

        if use_lighting:
            self.light_source_position = (random.random(), random.random())  # Normalized

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def _sample_normalized_value(self, key: str) -> float:
        lower, upper = self._continuous_bounds[key]
        return random.uniform(lower, upper)

    def _draw_random_parameters(self):
        norm_pos_x = self._sample_normalized_value('x_pos')
        norm_pos_y = self._sample_normalized_value('y_pos')
        norm_size = self._sample_normalized_value('size')
        norm_rotation = self._sample_normalized_value('rotation')
        return {
            'pos_x': norm_pos_x * self.image_shape[0],
            'pos_y': norm_pos_y * self.image_shape[1],
            'size': norm_size * min(self.image_shape),
            'rotation': norm_rotation * 360.0,
            'norm_pos_x': norm_pos_x,
            'norm_pos_y': norm_pos_y,
            'norm_size': norm_size,
            'norm_rotation': norm_rotation,
        }

    def generate_random_image(self, fixed_num_objects=False):
        num_objects = self.max_objects if fixed_num_objects else random.randint(1, self.max_objects)

        # Initialize with background
        if self.realism_mode == 'realistic':
            # Smooth gradient background
            background = self._generate_gradient_background()
            image = background.copy()
        else:
            image = np.zeros(self.image_shape, dtype=np.float32)

        # Add background noise
        if self.background_noise_level > 0:
            noise = np.random.normal(0, self.background_noise_level, size=self.image_shape)
            image = image + noise

        self.labels_list = []
        labels = {}
        bounding_boxes = []

        for obj_idx in range(num_objects):
            shape = random.choice(self.shapes)
            params = self._draw_random_parameters()
            pos_x, pos_y, size, rotation = (
                params['pos_x'], params['pos_y'], params['size'], params['rotation']
            )

            # Retry on overlap
            attempts = 0
            while self.check_overlap(image, shape, pos_x, pos_y, size) and attempts < 50:
                params = self._draw_random_parameters()
                pos_x, pos_y, size, rotation = (
                    params['pos_x'], params['pos_y'], params['size'], params['rotation']
                )
                attempts += 1

            min_corner, max_corner = self.compute_bounding_box(shape, size, pos_x, pos_y)
            bounding_boxes.append((min_corner, max_corner))

            # Build labels
            if self.use_lighting:
                if self.labels_as_text:
                    geometry_one_hot = self.geometry_one_hot[shape].tolist()
                    self.labels_list.extend(
                        geometry_one_hot + [
                            self.light_source_position[0],
                            self.light_source_position[1],
                            params['norm_pos_x'],
                            params['norm_pos_y'],
                            params['norm_size'],
                            params['norm_rotation'],
                        ]
                    )
                else:
                    labels[obj_idx] = {
                        'shape': shape,
                        'light_x': self.light_source_position[0],
                        'light_y': self.light_source_position[1],
                        'pos_x': params['norm_pos_x'],
                        'pos_y': params['norm_pos_y'],
                        'size': params['norm_size'],
                        'rotation': params['norm_rotation'],
                    }
            else:
                if self.labels_as_text:
                    geometry_one_hot = self.geometry_one_hot[shape].tolist()
                    self.labels_list.extend(
                        geometry_one_hot + [
                            params['norm_pos_x'],
                            params['norm_pos_y'],
                            params['norm_size'],
                            params['norm_rotation'],
                        ]
                    )
                else:
                    labels[obj_idx] = {
                        'shape': shape,
                        'pos_x': params['norm_pos_x'],
                        'pos_y': params['norm_pos_y'],
                        'size': params['norm_size'],
                        'rotation': params['norm_rotation'],
                    }

            # Generate shape with realism enhancements
            shape_layer = self._create_shape(shape, pos_x, pos_y, size, rotation)

            # Add to image (max operation preserves shape boundaries)
            image = np.maximum(image, shape_layer)

        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)

        return image, labels, bounding_boxes

    def _create_shape(self, shape, pos_x, pos_y, size, rotation):
        """Create shape with progressive realism based on mode."""
        if shape == 'circle':
            mask = self._create_circle(pos_x, pos_y, size)
        elif shape == 'square':
            mask = self._create_square(pos_x, pos_y, size, rotation)
        elif shape == 'triangle':
            mask = self._create_triangle(pos_x, pos_y, size, rotation)
        elif shape == 'star':
            mask = self._create_star(pos_x, pos_y, size, rotation)

        # Apply realism enhancements
        if self.realism_mode == 'binary':
            return mask  # Original binary

        elif self.realism_mode == 'textured':
            # Anti-aliasing
            if self.antialiasing:
                mask = gaussian_filter(mask, sigma=0.8)

            # Add texture
            texture = self._generate_texture(mask)
            mask = mask * texture

        elif self.realism_mode == 'realistic':
            # Anti-aliasing
            if self.antialiasing:
                mask = gaussian_filter(mask, sigma=1.2)

            # Add texture
            texture = self._generate_texture(mask)
            mask = mask * texture

            # Add shadow if lighting enabled
            if self.use_lighting:
                shadow = self._generate_shadow(mask, pos_x, pos_y)
                mask = np.maximum(mask, shadow * 0.3)  # Shadows are darker

            # Add subtle gradient for 3D effect
            gradient = self._generate_radial_gradient(pos_x, pos_y, size)
            mask = mask * (0.7 + 0.3 * gradient)

        return mask

    def _create_circle(self, pos_x, pos_y, size):
        """Anti-aliased circle using distance field."""
        y, x = np.ogrid[:self.image_shape[0], :self.image_shape[1]]
        dist = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
        radius = size / 2

        if self.antialiasing and self.realism_mode != 'binary':
            # Smooth transition at boundary
            mask = np.clip(radius - dist + 0.5, 0, 1)
        else:
            mask = (dist < radius).astype(np.float32)

        return mask

    def _create_square(self, pos_x, pos_y, size, rotation):
        mask = np.zeros(self.image_shape, dtype=np.float32)
        half_size = size / 2
        coords = np.array([
            [pos_x - half_size, pos_y - half_size],
            [pos_x + half_size, pos_y - half_size],
            [pos_x + half_size, pos_y + half_size],
            [pos_x - half_size, pos_y + half_size]
        ])
        coords = self._rotate_coords(coords, rotation)
        rr, cc = self._draw_polygon(coords)

        if len(rr) > 0:
            rr = np.clip(rr, 0, self.image_shape[0] - 1)
            cc = np.clip(cc, 0, self.image_shape[1] - 1)
            mask[rr, cc] = 1.0

        return mask

    def _create_triangle(self, pos_x, pos_y, size, rotation):
        mask = np.zeros(self.image_shape, dtype=np.float32)
        height = size * (np.sqrt(3) / 2)
        coords = np.array([
            [pos_x, pos_y - 2 / 3 * height],
            [pos_x - size / 2, pos_y + height / 3],
            [pos_x + size / 2, pos_y + height / 3]
        ])
        coords = self._rotate_coords(coords, rotation)
        rr, cc = self._draw_polygon(coords)

        if len(rr) > 0:
            rr = np.clip(rr, 0, self.image_shape[0] - 1)
            cc = np.clip(cc, 0, self.image_shape[1] - 1)
            mask[rr, cc] = 1.0

        return mask

    def _create_star(self, pos_x, pos_y, size, rotation):
        mask = np.zeros(self.image_shape, dtype=np.float32)
        coords = self._draw_star_coords(pos_x, pos_y, size, rotation)
        rr, cc = self._draw_polygon(coords)

        if len(rr) > 0:
            rr = np.clip(rr, 0, self.image_shape[0] - 1)
            cc = np.clip(cc, 0, self.image_shape[1] - 1)
            mask[rr, cc] = 1.0

        return mask

    def _draw_star_coords(self, cx, cy, size, rotation):
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        angles += rotation * (np.pi / 180.0)
        outer_radius = size / 2
        inner_radius = outer_radius / 2
        radii = np.array([outer_radius, inner_radius] * 5)
        x = cx + radii * np.cos(angles)
        y = cy + radii * np.sin(angles)
        return np.vstack((x, y)).T

    def _draw_polygon(self, coords):
        """Rasterize polygon using matplotlib."""
        rr, cc = [], []
        poly = Polygon(coords)

        # Get bounding box to reduce computation
        min_x = max(0, int(np.floor(coords[:, 0].min())))
        max_x = min(self.image_shape[1], int(np.ceil(coords[:, 0].max())) + 1)
        min_y = max(0, int(np.floor(coords[:, 1].min())))
        max_y = min(self.image_shape[0], int(np.ceil(coords[:, 1].max())) + 1)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if poly.contains_point([x, y]):
                    cc.append(y)
                    rr.append(x)

        return np.array(cc), np.array(rr)

    def _generate_texture(self, mask):
        """Generate procedural texture for shape."""
        # Perlin-like noise texture
        texture = np.random.rand(*self.image_shape) * 0.3 + 0.7
        texture = gaussian_filter(texture, sigma=3.0)

        # Apply only where mask is non-zero
        texture = np.where(mask > 0.1, texture, 1.0)
        return texture

    def _generate_shadow(self, mask, pos_x, pos_y):
        """Generate soft shadow based on light source."""
        light_x = self.light_source_position[0] * self.image_shape[1]
        light_y = self.light_source_position[1] * self.image_shape[0]

        # Shadow offset opposite to light direction
        dx = pos_x - light_x
        dy = pos_y - light_y
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        offset_x = int(10 * dx / norm)
        offset_y = int(10 * dy / norm)

        # Shift and blur mask for shadow
        shadow = np.zeros_like(mask)
        h, w = self.image_shape

        if 0 <= pos_y + offset_y < h and 0 <= pos_x + offset_x < w:
            sy_start = max(0, offset_y)
            sy_end = min(h, h + offset_y)
            sx_start = max(0, offset_x)
            sx_end = min(w, w + offset_x)

            my_start = max(0, -offset_y)
            my_end = min(h, h - offset_y)
            mx_start = max(0, -offset_x)
            mx_end = min(w, w - offset_x)

            shadow[sy_start:sy_end, sx_start:sx_end] = mask[my_start:my_end, mx_start:mx_end]

        shadow = gaussian_filter(shadow, sigma=5.0)
        return shadow

    def _generate_radial_gradient(self, cx, cy, size):
        """Generate radial gradient for 3D shading effect."""
        y, x = np.ogrid[:self.image_shape[0], :self.image_shape[1]]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        gradient = np.clip(1.0 - dist / (size * 0.7), 0, 1)
        return gradient

    def _generate_gradient_background(self):
        """Generate smooth gradient background."""
        y, x = np.ogrid[:self.image_shape[0], :self.image_shape[1]]
        gradient = 0.05 + 0.1 * (y / self.image_shape[0])
        return gradient

    def _rotate_coords(self, coords, angle):
        """Rotate coordinates around their centroid."""
        cx, cy = coords.mean(axis=0)
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        coords_shifted = coords - np.array([cx, cy])
        coords_rotated = coords_shifted @ rotation_matrix.T
        coords_rotated += np.array([cx, cy])
        return coords_rotated

    def compute_bounding_box(self, shape, size, pos_x, pos_y):
        half_size = size / 2
        margin = 0.02 * self.image_shape[0]
        min_corner = np.array([pos_x - half_size - margin, pos_y - half_size - margin])
        max_corner = np.array([pos_x + half_size + margin, pos_y + half_size + margin])
        return min_corner, max_corner

    def check_overlap(self, image, shape, pos_x, pos_y, size):
        min_corner, max_corner = self.compute_bounding_box(shape, size, pos_y, pos_x)
        min_corner = np.maximum(min_corner, 0).astype(int)
        max_corner = np.minimum(max_corner, self.image_shape).astype(int)

        if np.any(image[min_corner[0]:max_corner[0], min_corner[1]:max_corner[1]] > 0.5):
            return True
        return False

    def save_image(self, image, labels, idx):
        image_filename = os.path.join(self.image_dir, f'image_{idx}.png')
        plt.imsave(image_filename, image, cmap='gray', vmin=0, vmax=1)

        if self.labels_as_text:
            label_filename = os.path.join(self.label_dir, f'labels_{idx}.txt')
            with open(label_filename, 'w') as f:
                f.write(str(self.labels_list))
        else:
            label_filename = os.path.join(self.label_dir, f'labels_{idx}.json')
            with open(label_filename, 'w') as f:
                json.dump(labels, f)

    def generate_and_save_images(self, num_images, plot_images=False):
        for i in range(num_images):
            if i % 100 == 0:
                print(f'Generating image {i}/{num_images}...')
            image, labels, bounding_boxes = self.generate_random_image(self.fixed_num_objects)
            self.save_image(image, labels, i)
            if plot_images and i < 5:
                self.visualize_image(image, bounding_boxes)

    def visualize_image(self, image, bounding_boxes):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        for bbox in bounding_boxes:
            min_corner, max_corner = bbox
            rect = plt.Rectangle(min_corner, max_corner[0] - min_corner[0],
                                 max_corner[1] - min_corner[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.title(f'Mode: {self.realism_mode}')
        plt.tight_layout()
        plt.show()


# Usage example
if __name__ == '__main__':
    # Phase 1: Binary (your current setup)
    # gen_binary = EnhancedGeometryGenerator(
    #     realism_mode='binary',
    #     labels_as_text=True
    # )
    # gen_binary.generate_and_save_images(1000)

    # Phase 2: Textured (intermediate)
    gen_textured = EnhancedGeometryGenerator(
        realism_mode='binary',
        max_objects=2,
        background_noise_level=0.05,
        labels_as_text=True
    )
    gen_textured.generate_and_save_images(1000, plot_images=True)

    # Phase 3: Realistic (final before real images)
    # gen_realistic = EnhancedGeometryGenerator(
    #     realism_mode='realistic',
    #     background_noise_level=0.1,
    #     use_lighting=True,
    #     labels_as_text=True
    # )
    # gen_realistic.generate_and_save_images(1000)