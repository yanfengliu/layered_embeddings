from PIL import Image, ImageDraw
from IPython.display import clear_output
from scipy.ndimage.filters import gaussian_filter

from keras import regularizers
from keras.regularizers import l2
from keras.engine.topology import Layer
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical, plot_model
from keras.models import Sequential, Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, Dense, Reshape, Lambda, LeakyReLU, BatchNormalization, Concatenate

import os
import cv2
import sys
import math
import time
import scipy
import colorsys
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt


def semantic_loss(y_true, y_pred):
    """
    Semantic loss is simply categorical cross entropy loss applied to both front class mask 
    and back class mask. 
    """

    # ******************* CHANGE THESE VARS ACCORDING TO YOUR TASK ********************
    class_num = 46 # includes background
    # *********************************************************************************

    front_class_mask        = y_true[:, :, :, 0]
    back_class_mask         = y_true[:, :, :, 1]

    front_class_pred        = y_pred[:, :, :, :class_num]
    back_class_pred         = y_pred[:, :, :, class_num:(2*class_num)]

    front_class_mask = tf.cast(front_class_mask, tf.int32)
    front_class_mask_one_hot = tf.one_hot(front_class_mask, class_num)

    back_class_mask = tf.cast(back_class_mask, tf.int32)
    back_class_mask_one_hot = tf.one_hot(back_class_mask, class_num)

    front_class_mask_flat   = tf.reshape(front_class_mask_one_hot, shape=(-1, class_num))
    front_class_pred_flat   = tf.reshape(front_class_pred, shape=(-1, class_num))
    back_class_mask_flat    = tf.reshape(back_class_mask_one_hot, shape=(-1, class_num))
    back_class_pred_flat    = tf.reshape(back_class_pred, shape=(-1, class_num))

    cost1 = K.mean(K.categorical_crossentropy(
        tf.cast(front_class_mask_flat, tf.float32), tf.cast(front_class_pred_flat, tf.float32)))
    cost2 = K.mean(K.categorical_crossentropy(
        tf.cast(back_class_mask_flat, tf.float32), tf.cast(back_class_pred_flat, tf.float32)))
    cost = cost1 + cost2
    cost = tf.reshape(cost, [-1])

    return cost


def avoid_zero(x, threshold):
    """
    Detect and move a tiny value above the defined threhsold.

    Inputs:
    =======
    x: float -- value to be thresholded
    threshold: float -- must be positive

    Outputs:
    ========
    Thresholded value

    """
    if (abs(x) < threshold):
        if x < 0:
            return -threshold
        else:
            return threshold


class Evaluator:
    """
    A self-contained evaluator class that calculates performance metrics for the model. Heavily 
    borrowed code from COCO Amodal API. 
    """
    def __init__(self, gt, pred, pred_scores, parameters):
        self.gt_all = gt
        self.pred_all = pred
        self.pred_scores_all = pred_scores
        self.results = []
        self.eval = {}
        self.img_ids                = parameters["img_ids"]
        self.class_ids              = parameters["class_ids"]
        self.max_detection          = parameters["max_detection"]
        self.iou_thresholds         = parameters["iou_thresholds"]
        self.recall_thresholds      = parameters["recall_thresholds"]


    def evaluate_img(self, class_id, img_id, max_detection):
        # read ground truth and prediction data per image
        gt = self.gt_all[img_id][class_id]
        pred = self.pred_all[img_id][class_id]
        pred_scores = self.pred_scores_all[img_id][class_id]
        # sort prediction and score by score (descending)
        sort_idx = np.argsort(pred_scores)[::-1]
        pred_sorted = [x for _,x in sorted(zip(sort_idx,pred))]
        pred_scores_sorted = [x for _,x in sorted(zip(sort_idx,pred_scores))]
        # cut off at top N = max_detection results
        pred_sorted = pred_sorted[:max_detection]
        pred_scores_sorted = pred_scores_sorted[:max_detection]

        T = len(self.iou_thresholds)    # number of iou thresholds
        G = len(gt)                     # number of gt 
        P = len(pred_sorted)            # number of pred 
        
        pred_with_matched_gt = np.zeros((T, P)) - 1
        gt_with_matched_pred = np.zeros((T, G)) - 1

        for t, threshold in enumerate(self.iou_thresholds):

            # Note that in the traditional modal instance segmentation, if the iou threshold >= 0.5
            # then a prediction physically cannot match with more than 1 gt; however, this is not true 
            # in amodal instance segmentation because gts can overlap, just like predictions. So it 
            # is possible that multiple gts are overlapping with 1 pred, or multiple preds are matched 
            # with 1 gt. 

            # To address this, we follow "What Makes Effective Detection Proposals", where the authors 
            # specify that they compute a bipartite matching to assign preds to gts using a greedy 
            # algorithm instead of the optimal Hungarian algorithm. Meaning that if a prediction is 
            # matched to a gt that has the highest IoU with the pred among all gt, then that gt is no 
            # longer considered for other matchings, even if another prediction could potentially have 
            # a higher IoU with it. Note that this has to be done after sorting the predictions by score. 


            for i in range(P):
                matched_gt_idx = -1
                for j in range(G):
                    iou_threshold = threshold
                    iou = IoU(pred_sorted[i], gt[j])
                    if (matched_gt_idx > -1):
                        continue
                    if (iou > iou_threshold):
                        iou_threshold = iou
                        matched_gt_idx = j
                if (matched_gt_idx > -1):
                    pred_with_matched_gt[t, i] = matched_gt_idx
                    gt_with_matched_pred[t, matched_gt_idx] = i

            matches = {}
            matches["pred_with_matched_gt"] = pred_with_matched_gt
            matches["gt_with_matched_pred"] = gt_with_matched_pred
            matches["score_list"] = pred_scores_sorted

        return matches


    def evaluate(self):
        # self.results has dimension (num_class, num_iou_threshold, num_img)
        for class_id in self.class_ids: 
            img_results = []
            for img_id in self.img_ids:
                img_result = self.evaluate_img(class_id, img_id, self.max_detection)
                img_results.append(img_result)
            self.results.append(img_results)
    

    def get_ap_ar(self):
        T = len(self.iou_thresholds)
        R = len(self.recall_thresholds)
        K = len(self.class_ids)
        # TODO: implement precision
        precision = -np.ones((T, R, K))
        recall = -np.ones((T, K))
        for class_id in self.class_ids:
            scores = np.concatenate([m["score_list"] for m in self.results[class_id]])
            pred_with_matched_gt = np.concatenate(
                [m["pred_with_matched_gt"] for m in self.results[class_id]], axis = 1)
            gt_with_matched_pred = np.concatenate(
                [m["gt_with_matched_pred"] for m in self.results[class_id]], axis = 1)
            # mergesort for fastest speed at the cost of extra memory 
            idx = np.argsort(scores, kind='mergesort')
            pred_with_matched_gt = pred_with_matched_gt[:, idx]
            tps = pred_with_matched_gt > -1
            fps = pred_with_matched_gt == -1
            tp_sum = np.cumsum(tps, axis = 1).astype(np.float)
            fp_sum = np.cumsum(fps, axis = 1).astype(np.float)
            num_gt = len(gt_with_matched_pred[0])

            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                num_pred = len(tp)
                rc = tp / num_gt
                pr = tp / (tp + fp)

                if num_pred:
                    recall[t, class_id] = rc[-1]
                else:
                    recall[t, class_id] = 0
        self.eval['recall'] = recall


def single_class_loss(y_true, y_pred):
    """
    Composite loss function for pixel embedding and occlusion mask prediciton. Improved
    upon the first version by combining both embedding outputs into one flattened table
    and adjust the masks according so that the embedding loss can be computed jointly. 

    Inputs:
    ======
    y_true: tensor -- not actually ground truth embedding. Used to sneak in mask and 
        weight matrices; 
        y_true = Concatenate(w, mask, tri_state_mask, visible_occ_mask, occ_mask)
    y_pred: tensor -- embedding and mask outputs from the network;
        y_pred = Concatenate(embedding, tri_state_mask, occlusion_embedding)

    Outputs:
    ========
    cost: tensor -- a float number that represents the composite loss; equals combined 
        discriminative embedding loss and mask categorical crossentropy loss. 
    """

    # hyperparameters
    delta_var = 0.5
    delta_d = 1.5

    # get stats
    embedding_dim = 12
    num_class = 3

    # unpack ground truth contents (SKIP visible occ mask)
    w = y_true[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    tri_state_mask_true = y_true[:, :, :, 2:5]
    # visible_occ_mask = y_true[:, :, :, 5] NOT USED IN LOSS FUNCTION
    occ_mask = y_true[:, :, :, 6]
    class_mask = y_true[:, :, :, 7:]

    # y_pred
    emb_pred = y_pred[:, :, :, :embedding_dim]
    tri_state_mask_pred = y_pred[:, :, :, embedding_dim:(embedding_dim + 3)]
    occ_emb_pred = y_pred[:, :, :, (embedding_dim + 3):((2*embedding_dim + 3))]
    class_pred = y_pred[:, :, :, -(num_class+1):]

    # get number of pixels and clusters (without background)
    num_cluster = tf.reduce_max(mask)
    num_cluster = tf.cast(num_cluster, tf.int32)

    # one-hot encoding for mask
    mask = tf.cast(mask, tf.int32)
    mask = mask - 1
    mask_one_hot = tf.one_hot(mask, num_cluster)
    occ_mask = tf.cast(occ_mask, tf.int32)
    occ_mask = occ_mask - 1
    occ_mask_one_hot = tf.one_hot(occ_mask, num_cluster)

    # flatten
    emb_pred_flat = tf.reshape(emb_pred, shape=(-1, embedding_dim))
    occ_emb_pred_flat = tf.reshape(occ_emb_pred, shape=(-1, embedding_dim))
    mask_one_hot_flat = tf.reshape(mask_one_hot, shape=(-1, num_cluster))
    w_flat = K.flatten(w)
    mask_flat = K.flatten(mask)
    occ_mask_one_hot_flat = tf.reshape(occ_mask_one_hot, shape=(-1, num_cluster))
    occ_mask_flat = K.flatten(occ_mask)

    # combine embeddings and masks
    combined_emb_flat = tf.concat((emb_pred_flat, occ_emb_pred_flat), axis=0)
    combined_mask_flat = tf.concat((mask_flat, occ_mask_flat), axis=0)
    combined_mask_one_hot_flat = tf.concat((mask_one_hot_flat, occ_mask_one_hot_flat), axis=0)

    # ignore background pixels
    non_background_idx = tf.greater(combined_mask_flat, -1)
    combined_emb_flat = tf.boolean_mask(combined_emb_flat, non_background_idx)
    combined_mask_flat = tf.boolean_mask(combined_mask_flat, non_background_idx)
    combined_mask_one_hot_flat = tf.boolean_mask(combined_mask_one_hot_flat, non_background_idx)

    # center count
    center_count = tf.reduce_sum(tf.cast(combined_mask_one_hot_flat, dtype=tf.float32), axis=0)

    # variance term
    centers = tf.matmul(
        tf.transpose(combined_emb_flat), tf.cast(combined_mask_one_hot_flat, dtype=tf.float32))
    centers = tf.divide(centers, center_count)
    gathered_center = tf.gather(centers, combined_mask_flat, axis=1)
    gathered_center_count = tf.gather(center_count, combined_mask_flat)
    combined_emb_t = tf.transpose(combined_emb_flat)
    var_dist = tf.norm(combined_emb_t - gathered_center, ord=1, axis=0) - delta_var
    # changed from soft hinge loss to hard cutoff
    variance_term = tf.square(tf.maximum(var_dist, 0))
    variance_term = tf.divide(variance_term, gathered_center_count)
    variance_term = tf.reduce_sum(variance_term) / tf.cast(num_cluster, tf.float32)

    # center distance term
    centers_row_buffer = tf.ones((embedding_dim, num_cluster, num_cluster))
    centers = tf.expand_dims(centers, axis=2)
    centers_row = tf.multiply(centers_row_buffer, centers)
    centers_col = tf.transpose(centers_row, perm=[0, 2, 1])
    dist_matrix = centers_row - centers_col
    idx2 = tf.ones((num_cluster, num_cluster))
    diag = tf.ones((1, num_cluster))
    diag = tf.reshape(diag, [-1])
    idx2 = idx2 - tf.diag(diag)
    idx2 = tf.cast(idx2, tf.bool)
    idx2 = K.flatten(idx2)
    dist_matrix = tf.reshape(dist_matrix, [embedding_dim, -1])
    dist_matrix = tf.transpose(dist_matrix)
    sampled_dist = tf.boolean_mask(dist_matrix, idx2)
    distance_term = tf.square(tf.maximum(
        2 * delta_d - tf.norm(sampled_dist, ord=1, axis=1), 0))
    distance_term = tf.reduce_sum(
        distance_term) / tf.cast(num_cluster * (num_cluster - 1) + 1, tf.float32)

    # regularization term
    regularization_term = tf.reduce_mean(tf.norm(tf.squeeze(centers), ord=1, axis=0))

    # filter tri-state mask pred and gt with weight matrix
    w_flat = w_flat > 0
    tri_state_mask_pred_flat = tf.reshape(tri_state_mask_pred, [-1, 3])
    tri_state_mask_pred_flat = tf.boolean_mask(tri_state_mask_pred_flat, w_flat)
    tri_state_mask_true_flat = tf.reshape(tri_state_mask_true, [-1, 3])
    tri_state_mask_true_flat = tf.boolean_mask(tri_state_mask_true_flat, w_flat)

    # sum up terms
    cost1 = variance_term + distance_term + 0.01 * regularization_term
    cost2 = K.mean(K.categorical_crossentropy(tri_state_mask_true_flat, tri_state_mask_pred_flat), axis=-1)
    cost3 = K.mean(K.categorical_crossentropy(class_mask, class_pred), axis = -1)
    cost = cost1 + cost2 + cost3
    cost = tf.reshape(cost, [-1])

    return cost


def L2_norm(x):
    """
    Custom function for the L2_normalization layer
    """

    return K.l2_normalize(x, axis=3)


def RNN_mean_shift(x):
    """
    Implements recurrent mean shift based on https://arxiv.org/abs/1712.08273

    Inputs:
    =======
    x: keras model
    num_pixels: int -- total number of pixels in x
    margin: float -- margin value for cosine similarity cost
    eta: float -- step size in the updating stage
    embedding_dim: int -- number of channels in the output embedding

    Outputs:
    ========
    x: keras model
    """

    num_pixels = 56 * 56
    embedding_dim = 6
    margin = 0.8
    eta = 1.0
    iteration = 5

    x_shape = K.int_shape(x)
    x_shape = np.array(x_shape)
    x_shape[0] = 1
    x = K.reshape(x, shape=(-1, embedding_dim))
    N = num_pixels
    delta = 3 / (1 - margin)
    for itr in range(iteration):
        S = tf.matmul(x, x, transpose_b=True)
        G = tf.exp(delta * S)
        ones = tf.ones(shape=(N, 1))
        d = tf.matmul(G, ones)
        q = tf.divide(1.0, d)
        q = tf.reshape(q, [-1])
        temp = tf.matmul(G, tf.diag(q))
        P = (1 - eta) * tf.eye(N) + eta * temp
        x = tf.matmul(P, x)
        x = K.l2_normalize(x, axis=1)
    x = K.reshape(x, x_shape)

    return x