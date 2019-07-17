import argparse
import math
import os
import pickle
import random
import sys
import time
import warnings
import copy
import json

sys.path.append('C:/Users/yliu60/Documents/GitHub/amodalAPI/PythonAPI/pycocotools')

import cv2
import numpy as np
import pylab
import scipy
import scipy.ndimage.interpolation as interpolation
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from collections import defaultdict

import embedding_model as em
import post_processing as pp
import params

warnings.filterwarnings('ignore')
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


def normalize(x):
    """
    Normalize input to be zero mean and divide it by its global maximum value. 
    """

    x = x - np.min(x, keepdims=False)
    x = x / (np.max(x, keepdims=False) + 1e-10)
    return np.copy(x)


def intersection(mask1, mask2):
    return np.sum((mask1 > 0) & (mask2 > 0))


def union(mask1, mask2):
    return np.sum((mask1 > 0) | (mask2 > 0))


def IoU(mask1, mask2):
    mask1 = np.squeeze(mask1)
    mask2 = np.squeeze(mask2)
    i = intersection(mask1, mask2)
    u = union(mask1, mask2)
    return i / u


def totuple(a):
    """
    Convert a numpy array to a tuple of tuples in the format of [(), (), ...]
    """
    try:
        return [tuple(i) for i in a]
    except TypeError:
        return a


def vector_angle(v1, v2):
    """
    Return the angle between two vectors. 
    """
    cosine = (np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    radian = np.arccos(cosine)
    angle = radian * 57.2958
    angle = abs(angle)
    return angle


def consecutive_integer(mask):
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


def add_xy(image):
    side = image.shape[0]
    temp = np.zeros((side, side, 2))
    for i in range(side):
        temp[i, :, 0] = i
    for j in range(side):
        temp[:, j, 1] = j
    temp = normalize(temp)
    image[:, :, 1:3] = temp
    return image


def align_instance(y):
    # re-align instance labels
    mask            = y[:, :, 2]
    occ_mask_temp   = y[:, :, 3]
    mask, change_log = consecutive_integer(mask)
    occ_mask = np.zeros(occ_mask_temp.shape)
    for i in range(len(change_log)):
        occ_mask[occ_mask_temp == change_log[i]] = i
    y[:, :, 2]   = mask
    y[:, :, 3]   = occ_mask

    return y


def augment_data(image_info, y):
    image = image_info['image']
    original_size = image_info['original_size']
    height = original_size[0]
    width = original_size[1]
    y = np.squeeze(y)
    y = copy.deepcopy(y)
    DR = y.shape[1]

    # left right flip
    r = np.random.random()
    if r > 0.5:
        image = image[:, ::-1, :]
        y = y[:, ::-1, :]
        
    # rotate
    angle = np.random.randint(low=-15, high=15)
    image = interpolation.rotate(image, angle, order=0, prefilter=False)
    y = interpolation.rotate(y, angle, order=0, prefilter=False)

    y = align_instance(y)
    
    # TODO: scaling

    # shift
    x_shift = np.random.randint(low = np.round(-0.2 * width), high = np.round(0.2 * width))
    y_shift = np.random.randint(low = np.round(-0.2 * height), high = np.round(0.2 * height))
    image = interpolation.shift(image, [x_shift, y_shift, 0], order=0, prefilter=False)

    # resizing y to the size of the image before shifting so that they are aligned
    image_shifted_size = image.shape[1]
    y_big = resize(y, [image_shifted_size, image_shifted_size], order=0, mode='constant', preserve_range=True)
    y_big = interpolation.shift(y_big, [x_shift, y_shift, 0], order=0, prefilter=False)
    y_original_size = y.shape[1]
    y = resize(y_big, [y_original_size, y_original_size], order=0, mode='constant', preserve_range=True)
    
    y = align_instance(y)

    # resizing back to original
    image = resize(image, original_size, order=0, mode='constant', preserve_range=True)
    y = resize(y, (DR, DR), order=0, mode='constant', preserve_range=True)
    
    y = align_instance(y)
    
    image_info_copy = copy.deepcopy(image_info)
    image_info_copy['image'] = image
    
    y = np.expand_dims(y, axis=0)

    return image_info_copy, y


#Class to represent a graph 
class Graph: 
    def __init__(self,vertices): 
        self.graph = defaultdict(list) #dictionary containing adjacency List 
        self.V = vertices #No. of vertices 
  
    # function to add an edge to graph 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    # A recursive function used by topologicalSort 
    def topologicalSortUtil(self,v,visited,stack): 
  
        # Mark the current node as visited. 
        visited[v] = True
  
        # Recur for all the vertices adjacent to this vertex 
        for i in self.graph[v]: 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 
  
        # Push current vertex to stack which stores result 
        stack.insert(0,v) 
  
    # The function to do Topological Sort. It uses recursive  
    # topologicalSortUtil() 
    def topologicalSort(self): 
        # Mark all the vertices as not visited 
        visited = [False]*self.V
        stack =[] 
  
        # Call the recursive helper function to store Topological 
        # Sort starting from all vertices one by one 
        for i in range(self.V): 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 
  
        return stack