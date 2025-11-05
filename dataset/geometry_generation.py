import os
import random
from typing import Optional

import numpy as np
import json
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from config.geometry import GeometryLabelSchema


class RandomImageGenerator:
    def __init__(
        self,
        image_shape=(256, 256),
        max_objects=5,
        fixed_num_objects=True,
        labels_as_text=False,
        use_lighting=False,
        geometry_label_schema: Optional[GeometryLabelSchema] = None,
    ):
        self.image_shape = image_shape
        self.max_objects = max_objects
        self.fixed_num_objects = fixed_num_objects
        self.shapes = ['circle', 'square', 'triangle', 'star']

        self.geometry_one_hot = {shape: np.eye(len(self.shapes))[i] for i, shape in enumerate(self.shapes)}
        self.geometry_label_schema = geometry_label_schema or GeometryLabelSchema()
        self._continuous_bounds = self.geometry_label_schema.continuous_bounds
        self.image_dir = f'./Geometries_{max_objects}_2D/images'
        self.label_dir = f'./Geometries_{max_objects}_2D/labels'
        self.labels_as_text = labels_as_text
        self.use_lighting = use_lighting
        if use_lighting:
            self.light_source_position = (random.randint(0, image_shape[0]), random.randint(0, image_shape[1]))
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
        if fixed_num_objects:
            num_objects = self.max_objects
        else:
            num_objects = random.randint(1, self.max_objects)

        image = np.zeros(self.image_shape)
        self.labels_list = []
        labels = {}
        bounding_boxes = []

        for obj_idx in range(num_objects):
            i = 0
            shape = random.choice(self.shapes)
            params = self._draw_random_parameters()
            pos_x, pos_y, size, rotation = (
                params['pos_x'],
                params['pos_y'],
                params['size'],
                params['rotation'],
            )

            # pos_x = random.uniform(0.50, 0.50) * self.image_shape[0]
            # pos_y = random.uniform(0.50, 0.50) * self.image_shape[1]
            # size = random.uniform(0.10, 0.60) * min(self.image_shape)
            # rotation = random.randint(0, 0)

            while self.check_overlap(image, shape, pos_x, pos_y, size):
                i += 1
                print('Overlap detected. Setting new position...')
                params = self._draw_random_parameters()
                pos_x, pos_y, size, rotation = (
                    params['pos_x'],
                    params['pos_y'],
                    params['size'],
                    params['rotation'],
                )
                # if i > 50:
                #     # Just plot the image with the failed check overlap
                #     min_corner, max_corner = self.compute_bounding_box(shape, size, pos_x, pos_y)
                #     bounding_boxes.append((min_corner, max_corner))
                #     self.visualize_image(image, bounding_boxes, plot_bounding_boxes=True)

            min_corner, max_corner = self.compute_bounding_box(shape, size, pos_x, pos_y)

            bounding_boxes.append((min_corner, max_corner))

            if self.use_lighting:
                if self.labels_as_text:
                    geometry_one_hot = self.geometry_one_hot[shape].tolist()
                    self.labels_list.extend(
                        geometry_one_hot
                        + [
                            self.light_source_position[0] / self.image_shape[0],
                            self.light_source_position[1] / self.image_shape[0],
                            params['norm_pos_x'],
                            params['norm_pos_y'],
                            params['norm_size'],
                            params['norm_rotation'],
                        ]
                    )
                else:
                    labels[obj_idx] = {
                        'shape': shape,
                        'light_x': self.light_source_position[0] / self.image_shape[0],
                        'light_y': self.light_source_position[1] / self.image_shape[0],
                        'pos_x': params['norm_pos_x'],
                        'pos_y': params['norm_pos_y'],
                        'size': params['norm_size'],
                        'rotation': params['norm_rotation'],
                    }
            else:
                if self.labels_as_text:
                    geometry_one_hot = self.geometry_one_hot[shape].tolist()
                    self.labels_list.extend(
                        geometry_one_hot
                        + [
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

            print(f'Generating geometry {obj_idx} - {shape}...')

            if shape == 'circle':
                image += self.create_circle(pos_x, pos_y, size)
            elif shape == 'square':
                image += self.create_square(pos_x, pos_y, size, rotation)
            elif shape == 'triangle':
                image += self.create_triangle(pos_x, pos_y, size, rotation)
            elif shape == 'star':
                image += self.create_star(pos_x, pos_y, size, rotation)

        return image, labels, bounding_boxes

    def generate_image_from_condition(self, condition):
        """
        Generates an image based on the provided condition vector.

        Parameters:
            condition (list or array): A list containing the condition vector,
                                       with shape as a one-hot vector and other parameters normalized.

        Returns:
            np.array: Generated image based on the given condition.
        """
        # Decode the condition vector
        shape_type_one_hot = condition[:len(self.shapes)].numpy()  # Assuming the first 4 values are for shape type
        pos_x = condition[len(self.shapes)].item() * self.image_shape[0]  # Denormalize position x
        pos_y = condition[len(self.shapes) + 1].item() * self.image_shape[1]  # Denormalize position y
        size = condition[len(self.shapes) + 2].item() * min(self.image_shape)  # Denormalize size
        rotation = condition[len(self.shapes) + 3].item() * 360  # Denormalize rotation

        # Determine shape based on the one-hot encoding
        shape_idx = np.argmax(shape_type_one_hot)
        shape = self.shapes[shape_idx]

        # Initialize a blank image
        image = np.zeros(self.image_shape)
        bounding_boxes = []

        # Generate shape based on decoded condition
        if shape == 'circle':
            image += self.create_circle(pos_x, pos_y, size)
        elif shape == 'square':
            image += self.create_square(pos_x, pos_y, size, rotation)
        elif shape == 'triangle':
            image += self.create_triangle(pos_x, pos_y, size, rotation)
        elif shape == 'star':
            image += self.create_star(pos_x, pos_y, size, rotation)

        # Compute bounding box and add it to bounding_boxes list
        min_corner, max_corner = self.compute_bounding_box(shape, size, pos_x, pos_y)
        bounding_boxes.append((min_corner, max_corner))

        # Optionally save or visualize the image here, if needed
        return image, bounding_boxes

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

        if np.any(image[min_corner[0]:max_corner[0], min_corner[1]:max_corner[1]]):
            return True
        return False

    def create_circle(self, pos_x, pos_y, size):
        circle = np.zeros(self.image_shape, dtype=int)
        rr, cc = self.draw_circle(pos_x, pos_y, size / 2)
        circle[rr, cc] = 1
        return circle

    def draw_circle(self, cx, cy, radius):
        rr, cc = [], []
        for x in range(self.image_shape[0]):
            for y in range(self.image_shape[1]):
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    rr.append(x)
                    cc.append(y)
        return cc, rr

    def create_square(self, pos_x, pos_y, size, rotation):
        square = np.zeros(self.image_shape, dtype=int)
        half_size = size / 2
        coords = np.array([
            [pos_x - half_size, pos_y - half_size],
            [pos_x + half_size, pos_y - half_size],
            [pos_x + half_size, pos_y + half_size],
            [pos_x - half_size, pos_y + half_size]
        ])
        coords = self.rotate_coords(coords, rotation)
        rr, cc = self.draw_polygon(coords)
        square[cc, rr] = 2
        return square

    def create_triangle(self, pos_x, pos_y, size, rotation):
        triangle = np.zeros(self.image_shape, dtype=int)
        height = size * (np.sqrt(3) / 2)
        coords = np.array([
            [pos_x, pos_y - 2 / 3 * height],
            [pos_x - size / 2, pos_y + height / 3],
            [pos_x + size / 2, pos_y + height / 3]
        ])
        coords = self.rotate_coords(coords, rotation)
        rr, cc = self.draw_polygon(coords)
        triangle[cc, rr] = 3
        return triangle

    def create_star(self, pos_x, pos_y, size, rotation):
        star = np.zeros(self.image_shape, dtype=int)
        coords = self.draw_star(pos_x, pos_y, size, rotation)
        rr, cc = self.draw_polygon(coords)
        star[rr, cc] = 4
        return star

    def draw_star(self, cx, cy, size, rotation):
        # Drawing a simple 5-point star
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        angles += rotation * (np.pi / 180.0)
        outer_radius = size / 2
        inner_radius = outer_radius / 2
        radii = np.array([outer_radius, inner_radius] * 5)
        x = cx + radii * np.cos(angles)
        y = cy + radii * np.sin(angles)
        return np.vstack((y, x)).T

    def draw_polygon(self, coords):
        rr, cc = [], []
        poly = Polygon(coords)
        for x in range(self.image_shape[0]):
            for y in range(self.image_shape[1]):
                if poly.contains_point([x, y]):
                    rr.append(x)
                    cc.append(y)
        return rr, cc

    def calculate_centroid(self, coords):
        """Calculate centroid (center of mass) of a set of coordinates."""
        x_mean = np.mean(coords[:, 0])
        y_mean = np.mean(coords[:, 1])
        return x_mean, y_mean

    def rotate_coords(self, coords, angle):
        """Rotate coordinates around their centroid by a given angle."""
        cx, cy = self.calculate_centroid(coords)
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        coords_shifted = coords - np.array([cx, cy])
        coords_rotated = np.dot(coords_shifted, rotation_matrix)
        coords_rotated += np.array([cx, cy])
        return coords_rotated

    def save_image(self, image, labels, idx):
        # normalize image by dividing through length of classes
        image = image / len(self.shapes)
        image_filename = os.path.join(self.image_dir, f'image_{idx}.png')
        plt.imsave(image_filename, image, cmap='gray')
        if self.labels_as_text:
            label_filename = os.path.join(self.label_dir, f'labels_{idx}.txt')
            with open(label_filename, 'w') as f:
                f.write(str(self.labels_list))
        else:
            label_filename = os.path.join(self.label_dir, f'labels_{idx}.json')
            with open(label_filename, 'w') as f:
                json.dump(labels, f)

    def generate_and_save_images(self, num_images, plot_images=False, plot_bounding_boxes=True):
        for i in range(num_images):
            print(f'\n--- Generate image: {i} ---\n')
            image, labels, bounding_boxes = self.generate_random_image(self.fixed_num_objects)
            self.save_image(image, labels, i)
            if plot_images:
                self.visualize_image(image, bounding_boxes, plot_bounding_boxes=plot_bounding_boxes)

    def visualize_image(self, image, bounding_boxes, plot_bounding_boxes=True):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        if plot_bounding_boxes:
            for bbox in bounding_boxes:
                min_corner, max_corner = bbox
                rect = plt.Rectangle(min_corner, max_corner[0] - min_corner[0], max_corner[1] - min_corner[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()


# Example usage:
if __name__ == '__main__':
    generator = RandomImageGenerator(use_lighting=False, labels_as_text=True)
    generator.generate_and_save_images(1000, plot_images=False, plot_bounding_boxes=False)
