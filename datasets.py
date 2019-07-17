import random
import numpy as np 
import skimage.io as io
import os 
import copy
import pickle
import json
import sys

sys.path.append('C:/Users/yliu60/Documents/GitHub/amodalAPI/PythonAPI/pycocotools')
sys.path.append('C:/Users/Yanfeng Liu/Documents/GitHub/amodalAPI/PythonAPI/pycocotools')

from utils import consecutive_integer, Graph, normalize, IoU
from PIL import Image, ImageDraw
from keras.utils import to_categorical
from skimage.transform import resize
from IPython.display import clear_output
import matplotlib.image as mpimg
import shapes


def shapes_generator(params):
    while True:
        yield get_batch_image_and_gt(params)

def get_single_image_and_gt(params):
    image_info = shapes.get_shapes(params)

    image                 = image_info['image']
    mask                  = image_info['first_layer_mask']
    occ_mask              = image_info['occ_mask']
    class_mask            = image_info['class_mask']
    back_class_mask_small = image_info['back_class_mask']

    # convert image from [0, 1] to [-1, 1]
    image = image * 2 - 1
    image_info['image'] = image
    y = np.concatenate((class_mask, back_class_mask_small, mask, occ_mask), axis=-1)

    return image_info, y


def get_batch_image_and_gt(params):
    batch_size  = params.BATCH_SIZE
    side        = params.SIDE
    DF          = params.DOWNSAMPLE_FACTOR

    batch_image = np.zeros(shape=(batch_size, side, side, 3))
    batch_gt = np.zeros(shape=(batch_size, side//DF, side//DF, 4))
    for i in range(batch_size):
        image_info, gt = get_single_image_and_gt(params)
        batch_image[i, :, :, :] = image_info['image']
        batch_gt[i, :, :, :] = gt

    return batch_image, batch_gt

