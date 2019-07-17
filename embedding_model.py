import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.datasets import mnist
from keras.layers import BatchNormalization, Concatenate, Dense, Input, Lambda, LeakyReLU, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot

from model import Deeplabv3

def loss_with_embedding_dim(params):
    def multi_class_instance_embedding_loss(y_true, y_pred):

        # hyperparameters
        batch_size              = params.BATCH_SIZE
        delta_var               = params.DELTA_VAR
        delta_d                 = params.DELTA_D
        class_num               = params.CLASS_NUM
        embedding_dim           = params.EMBEDDING_DIM

        total_cost = 0
        # unpack ground truth contents
        for j in range(batch_size):
            front_class_mask        = y_true[j:j+1, :, :, 0]
            back_class_mask         = y_true[j:j+1, :, :, 1]
            front_mask              = y_true[j:j+1, :, :, 2]
            back_mask               = y_true[j:j+1, :, :, 3]

            # y_pred
            front_class_pred        = y_pred[j:j+1, :, :, :class_num]
            back_class_pred         = y_pred[j:j+1, :, :, class_num:(2*class_num)]
            front_emb               = y_pred[j:j+1, :, :, (2*class_num):(2*class_num + embedding_dim)]
            back_emb                = y_pred[j:j+1, :, :, (2*class_num + embedding_dim):]

            # get number of pixels and clusters (without background)
            num_cluster = tf.reduce_max(front_mask)
            num_cluster = tf.cast(num_cluster, tf.int32)

            # one-hot encoding for mask
            front_mask = tf.cast(front_mask, tf.int32)
            front_mask = front_mask - 1
            front_mask_one_hot = tf.one_hot(front_mask, num_cluster)

            back_mask = tf.cast(back_mask, tf.int32)
            back_mask = back_mask - 1
            back_mask_one_hot = tf.one_hot(back_mask, num_cluster)

            front_class_mask = tf.cast(front_class_mask, tf.int32)
            front_class_mask_one_hot = tf.one_hot(front_class_mask, class_num)

            back_class_mask = tf.cast(back_class_mask, tf.int32)
            back_class_mask_one_hot = tf.one_hot(back_class_mask, class_num)

            # flatten
            front_emb_flat          = tf.reshape(front_emb, shape=(-1, embedding_dim))
            back_emb_flat           = tf.reshape(back_emb, shape=(-1, embedding_dim))

            front_mask_one_hot_flat = tf.reshape(front_mask_one_hot, shape=(-1, num_cluster))
            front_mask_flat         = K.flatten(front_mask)

            back_mask_one_hot_flat  = tf.reshape(back_mask_one_hot, shape=(-1, num_cluster))
            back_mask_flat          = K.flatten(back_mask)

            front_class_mask_flat   = tf.reshape(front_class_mask_one_hot, shape=(-1, class_num))
            front_class_pred_flat   = tf.reshape(front_class_pred, shape=(-1, class_num))
            back_class_mask_flat    = tf.reshape(back_class_mask_one_hot, shape=(-1, class_num))
            back_class_pred_flat    = tf.reshape(back_class_pred, shape=(-1, class_num))

            # combine embeddings and masks
            combined_emb_flat           = tf.concat((front_emb_flat, back_emb_flat), axis=0)
            combined_mask_flat          = tf.concat((front_mask_flat, back_mask_flat), axis=0)
            combined_mask_one_hot_flat  = tf.concat((front_mask_one_hot_flat, back_mask_one_hot_flat), axis=0)

            # ignore background pixels
            non_background_idx          = tf.greater(combined_mask_flat, -1)
            combined_emb_flat           = tf.boolean_mask(combined_emb_flat, non_background_idx)
            combined_mask_flat          = tf.boolean_mask(combined_mask_flat, non_background_idx)
            combined_mask_one_hot_flat  = tf.boolean_mask(combined_mask_one_hot_flat, non_background_idx)

            # center count
            center_count = tf.reduce_sum(tf.cast(combined_mask_one_hot_flat, dtype=tf.float32), axis=0)

            # variance term
            embedding_sum_by_instance = tf.matmul(
                tf.transpose(combined_emb_flat), tf.cast(combined_mask_one_hot_flat, dtype=tf.float32))
            centers = tf.divide(embedding_sum_by_instance, center_count)
            gathered_center = tf.gather(centers, combined_mask_flat, axis=1)
            gathered_center_count = tf.gather(center_count, combined_mask_flat)
            combined_emb_t = tf.transpose(combined_emb_flat)
            var_dist = tf.norm(combined_emb_t - gathered_center, ord=1, axis=0) - delta_var
            # changed from soft hinge loss to hard cutoff
            var_dist_pos = tf.square(tf.maximum(var_dist, 0))
            var_dist_by_instance = tf.divide(var_dist_pos, gathered_center_count)
            variance_term = tf.reduce_sum(var_dist_by_instance) / tf.cast(num_cluster, tf.float32)

            # get instance to class mapping
            front_class_mask = tf.expand_dims(front_class_mask, axis=-1)
            filtered_class = tf.multiply(tf.cast(front_mask_one_hot, tf.float32), tf.cast(front_class_mask, tf.float32))
            instance_to_class = tf.reduce_max(filtered_class, axis = [0, 1, 2])


            def true_fn(num_cluster_by_class, centers_by_class):
                centers_row_buffer = tf.ones((embedding_dim, num_cluster_by_class, num_cluster_by_class))
                centers_by_class = tf.expand_dims(centers_by_class, axis=2)
                centers_row = tf.multiply(centers_row_buffer, centers_by_class)
                centers_col = tf.transpose(centers_row, perm=[0, 2, 1])
                dist_matrix = centers_row - centers_col
                idx2 = tf.ones((num_cluster_by_class, num_cluster_by_class))
                diag = tf.ones((1, num_cluster_by_class))
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
                    distance_term) / tf.cast(num_cluster_by_class * (num_cluster_by_class - 1) + 1, tf.float32)
                return distance_term


            def false_fn():
                return 0.0


            distance_term_total = 0.0
            # center distance term
            for i in range(3):
                class_idx = tf.equal(instance_to_class, i+1)
                centers_transpose = tf.transpose(centers)
                centers_by_class_transpose = tf.boolean_mask(centers_transpose, class_idx)
                centers_by_class = tf.transpose(centers_by_class_transpose)
                num_cluster_by_class = tf.reduce_sum(tf.cast(class_idx, tf.float32))
                num_cluster_by_class = tf.cast(num_cluster_by_class, tf.int32)
                distance_term_subtotal = tf.cond(num_cluster_by_class > 0, 
                                                lambda: true_fn(num_cluster_by_class, centers_by_class), 
                                                lambda: false_fn())
                distance_term_total += distance_term_subtotal
                
            # regularization term
            regularization_term = tf.reduce_mean(tf.norm(tf.squeeze(centers), ord=1, axis=0))

            # sum up terms
            cost1 = variance_term + distance_term_total + 0.01 * regularization_term
            cost2 = K.mean(K.categorical_crossentropy(
                tf.cast(front_class_mask_flat, tf.float32), tf.cast(front_class_pred_flat, tf.float32)))
            cost3 = K.mean(K.categorical_crossentropy(
                tf.cast(back_class_mask_flat, tf.float32), tf.cast(back_class_pred_flat, tf.float32)))
            cost = cost1 + cost2 + cost3
            cost = tf.reshape(cost, [-1])
            total_cost += cost
            
        total_cost = total_cost / batch_size

        return total_cost

    return multi_class_instance_embedding_loss


def embedding_module(x, num_filter, embedding_dim, weight_decay=1E-5):
    for i in range(int(len(num_filter))):
        x = Conv2D(num_filter[i], (3, 3),
                kernel_initializer="he_uniform",
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay))(x)

    x = Conv2D(embedding_dim, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)

    return x


def softmax_module(x, num_filter, num_class, weight_decay=1E-5):
    for i in range(int(len(num_filter))):
        x = Conv2D(num_filter[i], (3, 3),
                kernel_initializer="he_uniform",
                padding="same",
                activation="relu",
                kernel_regularizer=l2(weight_decay))(x)

    x = Conv2D(filters=num_class, 
               kernel_size=(3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               activation='softmax',
               kernel_regularizer=l2(weight_decay))(x)

    return x


def EmbeddingModel(params):
    side = params.SIDE
    deeplab_model       = Deeplabv3(input_shape = (side, side, 3), backbone = params.BACKBONE)
    inputs              = deeplab_model.input
    middle              = deeplab_model.get_layer(deeplab_model.layers[-3].name).output
    front_class         = softmax_module(middle, params.NUM_FILTER, params.CLASS_NUM)
    back_class          = softmax_module(middle, params.NUM_FILTER, params.CLASS_NUM)
    front_embedding     = embedding_module(middle, params.NUM_FILTER, params.EMBEDDING_DIM)
    back_embedding      = embedding_module(middle, params.NUM_FILTER, params.EMBEDDING_DIM)
    final_results       = Concatenate(axis=-1)([front_class,
                                                back_class,
                                                front_embedding, 
                                                back_embedding])
    embedding_model = Model(inputs = inputs, outputs = final_results)
    return embedding_model