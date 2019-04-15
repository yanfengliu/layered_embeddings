"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from PIL import Image, ImageDraw
from shapes import draw_ellipse, get_rect, totuple
from skimage.transform import resize
import matplotlib.pyplot as plt


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, rectangles, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, num_instances_per_class):
        self.num_instances_per_class = num_instances_per_class
        super(ShapesDataset, self).__init__()

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "circle")
        self.add_class("shapes", 2, "triangle")
        self.add_class("shapes", 3, "rectangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img = Image.new(mode='RGB', size=(info['height'], info['width']), color=(255, 255, 255))
        draw_img = ImageDraw.Draw(img)
        for shape, angle, xys, side in info['shapes']:
            draw_img = self.draw_shape(draw_img, shape, angle, xys, side, "image")
        image = np.asarray(img)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, angle, xys, side) in enumerate(info['shapes']):
            mask_template = Image.new(mode='I', size=(info['height'], info['width']), color=0)
            draw_mask = ImageDraw.Draw(mask_template)
            draw_mask = self.draw_shape(draw_mask, shape, angle, xys, side, "mask")
            mask[:, :, i] = np.asarray(mask_template) / 255.0
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        mask, class_ids = self.filter_invisible(mask, class_ids)
        return mask, class_ids.astype(np.int32)
    
    def consecutive_integer(self, mask):
        """
        Convert input mask into consecutive integer values, starting from 0. 
        If the background class is missing to start with, we manually inject a background pixel at [0, 0]
        so that the loss function will run properly. We realize that this is suboptimal and will explore 
        better solutions in the future. 
        """

        mask_buffer = np.zeros(mask.shape)
        if (0 not in np.unique(mask)):
            mask[0, 0] = 0
        mask_values = np.unique(mask)
        change_log = np.zeros(shape=(len(mask_values)))
        counter = 0
        for value in mask_values:
            mask_buffer[mask == value] = counter
            change_log[counter] = value
            counter += 1
        mask = mask_buffer.astype(int)
        return mask, change_log

    def filter_invisible(self, mask, class_ids):
        counter = 1
        side = mask.shape[0]
        all_instances = np.zeros((side, side))
        for i in range(mask.shape[-1]):
            all_instances[mask[:, :, i] == 1] = counter
            counter += 1
        all_instances, change_log = self.consecutive_integer(all_instances)
        num_layer = len(np.unique(all_instances))
        filtered_mask = np.zeros((side, side, num_layer-1))
        filtered_class_ids = np.zeros((num_layer-1,))
        for i in range(1, num_layer):
            filtered_mask[:, :, i-1] = mask[:, :, int(change_log[i])-1]
            filtered_class_ids[i-1] = class_ids[int(change_log[i])-1]

        return filtered_mask, filtered_class_ids.astype(np.int32)

    def draw_shape(self, draw_img, shape, angle, xys, side, image_or_mask):
        """Draws a shape from the given specs."""
        line_width = int(0.01 * side)
        x_shift, y_shift, _ = xys
        # 1 for circle
        if (shape == "circle"):
            radius = 0.25 * side
            x0 = x_shift
            y0 = y_shift
            x1 = x0 + radius
            y1 = y0 + radius
            # draw image
            draw_img = draw_ellipse(draw=draw_img, bbox=[x0, y0, x1, y1], linewidth=line_width, 
                        image_or_mask=image_or_mask, counter='white')
        else:
            # 2 for triangle
            if (shape == "triangle"):
                L = 0.3 * side
                corners = [[0, 0],
                        [L, 0], 
                        [0.5 * L, 0.866 * L],
                        [0, 0]]
                theta = (np.pi / 180.0) * angle
                R = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
                offset = np.array([x_shift, y_shift])
                corners = np.dot(corners, R) + offset

            # 3 for rectangle
            if (shape == "rectangle"):
                width = 0.1 * side
                height = 0.8 * side
                corners = get_rect(x=x_shift, y=y_shift, width=width, height=height, angle=angle)

            # get tuple version of the points
            shape_tuple = totuple(corners)
            # draw image
            draw_img.polygon(xy=shape_tuple, fill='white', outline=0)
            if image_or_mask == "image":
                draw_img.line(xy=shape_tuple, fill='black', width=line_width)
            elif image_or_mask == "mask":
                draw_img.line(xy=shape_tuple, fill='white', width=line_width)
            # draw ellipses around the corner to make lines look nice
            for point in shape_tuple:
                draw_img.ellipse((point[0] - 0.5*line_width, 
                                point[1] - 0.5*line_width, 
                                point[0] + 0.5*line_width, 
                                point[1] + 0.5*line_width), 
                                fill='black')

        return draw_img

    def random_shape(self, side):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (rectangle, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        angle = np.random.randint(0, 360)
        x = np.round((np.random.rand() * 0.8 + 0.1)*side)
        y = np.round((np.random.rand() * 0.8 + 0.1)*side)
        buffer = 20
        s = buffer
        return angle, (x, y, s)
        

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.

        This function has been modified to use np.random module to make the 
        same random decisions as shapes.get_new_shape(); in other words, given the 
        same seed, both functions will output the same image. 
        """

        bg_color = np.array([255, 255, 255])
        shapes = []
        N = self.num_instances_per_class
        shape_type_list = ["circle"]*N + ["triangle"]*N + ["rectangle"]*N
        np.random.shuffle(shape_type_list)

        side = height
        for i in range(3*N):
            color, xys = self.random_shape(side=side)
            shapes.append((shape_type_list[i], color, xys, side))
            
        return bg_color, shapes