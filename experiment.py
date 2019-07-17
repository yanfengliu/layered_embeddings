import os 
import sys 
import json

sys.path.append('C:/Users/yliu60/Documents/GitHub/amodalAPI/PythonAPI/pycocotools')
sys.path.append('C:/Users/Yanfeng Liu/Documents/GitHub/amodalAPI/PythonAPI/pycocotools')

import shapes 
import params as params_lib
import post_processing as pp
import embedding_model as em 
import numpy as np 
import matplotlib.pyplot as plt
import metrics_hist
from utils import normalize, augment_data
from shapes import get_shapes
from datasets import get_batch_image_and_gt, get_single_image_and_gt, shapes_generator
from IPython.display import clear_output
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam


class Experiment(object):
    def __init__(self):
        self.model_name                 = None
        self.backbone                   = None
        self.num_filter                 = None
        self.embedding_dim              = None
        self.xy_coordinates             = None
        self.num_shape_per_class        = None
        self.steps_per_epoch            = None
        self.val_sample_num             = None
        self.total_epochs               = None
        self.optimizer                  = None
        self.current_training_epoch     = None
        self.current_saved_epoch        = None
        self.lr                         = None
        self.lr_decay                   = None
        self.batch_size                 = None
        self.workers                    = None
        self.eth_mean_shift_threshold   = None
        self.delta_var                  = None
        self.delta_d                    = None
        self.verbose                    = None
        self.verbose                    = None
        self.use_gt                     = None
        self.use_seg                    = None
        self.semantic_only              = None
        self.data_aug                   = None
        self.features_pool              = []
        self.class_ids                  = []
    
    def prepare_val_gt(self):
        self.check_and_create_dir()
        self.init_dataset_params()
        self.build_gt_json()

    def start(self):
        self.check_and_create_dir()
        self.init_dataset_params()
        self.build_gt_json()
        self.init_metrics_history()
        self.build_model()
        self.get_current_epoch()
        self.adjust_optimizer()
        self.compile_model()
        self.load_last_saved_weights()
        self.select_eval_set('val_non_random')
        self.train_and_evaluate()

    def check_and_create_dir(self):
        prefix = self.model_name + '/'
        for feature_idx in range(len(self.features_pool)-1):
            feature = self.features_pool[feature_idx]
            prefix += str(feature) + '-'
        prefix += str(self.features_pool[-1]) + '/'

        self.image_dir          = 'images/'
        self.sample_cache_dir   = 'sample_cache/'
        self.dt_dir             = 'results/' + prefix
        self.model_dir          = 'models/'  + prefix
        self.gt_json_dir        = 'gt_json/' + self.model_name + '/' + str(self.num_shape_per_class) + '/'

        if not os.path.exists(self.dt_dir):
            os.makedirs(self.dt_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.gt_json_dir):
            os.makedirs(self.gt_json_dir)

    def init_dataset_params(self):
        self.params_train_random    = params_lib.ParamsTrainRandom(self)
        self.params_val_random      = params_lib.ParamsValRandom(self)
        self.params_val_non_random  = params_lib.ParamsValNonrandom(self)
        self.params_test_non_random = params_lib.ParamsTestNonrandom(self)
    
    def build_gt_json(self):
        np.random.seed(self.num_shape_per_class)
        shapes.get_shapes.idx = 0
        self.generate_dataset_json()

    def init_metrics_history(self):
        self.metrics_hist = metrics_hist.MetricsHist(self.dt_dir)
        self.metrics_hist.load()
    
    def build_model(self):
        self.model = em.EmbeddingModel(self.params_train_random)

    def get_current_epoch(self):
        model_list = os.listdir(self.model_dir)
        if model_list == []:
            self.current_saved_epoch = 0
        else:
            current_epoch = 0
            for model_name in model_list:
                temp_str = model_name.split('-')
                temp_str = temp_str[2]
                temp_str = temp_str.split('.')
                temp_str = temp_str[0]
                model_epoch = int(temp_str)
                if (model_epoch > current_epoch):
                    current_epoch = model_epoch
            self.current_saved_epoch =  current_epoch
        self.current_training_epoch = np.max(self.current_saved_epoch, 0)
    
    def adjust_optimizer(self):
        self.lr = self.lr / (1 + self.current_saved_epoch * self.lr_decay)
        self.optimizer = Adam(lr = self.lr, decay = self.lr_decay)

    def compile_model(self):
        self.model.compile(loss = em.loss_with_embedding_dim(self.params_train_random), 
                            optimizer = self.optimizer)
    
    def load_last_saved_weights(self):
        if self.current_saved_epoch:
            weight_path = self.model_dir + 'shapes-epoch-' + str(self.current_saved_epoch) + '.h5'
            print('Loading weights from {}'.format(weight_path))
            self.model.load_weights(weight_path)
 
    def select_eval_set(self, set_str):
        params_eval_dict = {
            'train_random': self.params_train_random,
            'val_random': self.params_val_random,
            'val_non_random': self.params_val_non_random,
            'test_non_random': self.params_test_non_random
        }
        self.params_eval = params_eval_dict[set_str]

    def train_and_evaluate(self):
        for j in range(self.current_saved_epoch+1, self.total_epochs):
            self.print_config()
            self.train()

            if (j%5 == 4):
                print('Saving model. DO NOT INTERRUPT.')
                save_string = self.model_dir + 'shapes-epoch-' + str(j) + '.h5'
                self.model.save(save_string)
                if (j>10):
                    shapes.get_shapes.idx = 0
                    metrics = self.evaluate()
                    self.metrics_hist.append(metrics)
                    self.update_average_weight()
                    self.metrics_hist.save()
                    clear_output()
                    self.metrics_hist.plot()
    
    def update_average_weight(self):
        avg_w, deeplabv3_avg_w, embedding_avg_w = self.get_average_weight()
        self.metrics_hist.avg_w = np.append(self.metrics_hist.avg_w, avg_w)
        self.metrics_hist.deeplabv3_avg_w = np.append(self.metrics_hist.deeplabv3_avg_w, deeplabv3_avg_w)
        self.metrics_hist.embedding_avg_w = np.append(self.metrics_hist.embedding_avg_w, embedding_avg_w)
    
    def get_average_weight(self):
        layers = self.model.layers
        deeplabv3_layers = layers[:-17]
        embedding_layers = layers[-17:]
        deeplabv3_weight_sum = 0
        embedding_weight_sum = 0
        deeplabv3_weight_count = 0
        embedding_weight_count = 0

        for layer in deeplabv3_layers:
            for filters in layer.get_weights():
                deeplabv3_weight_sum += np.sum(np.abs(filters.flatten()))
                deeplabv3_weight_count += len(filters.flatten())
        deeplabv3_avg_w = deeplabv3_weight_sum / deeplabv3_weight_count

        for layer in embedding_layers:
            for filters in layer.get_weights():
                embedding_weight_sum += np.sum(np.abs(filters.flatten()))
                embedding_weight_count += len(filters.flatten())
        embedding_avg_w = embedding_weight_sum / embedding_weight_count

        weight_sum = deeplabv3_weight_sum + embedding_weight_sum
        weight_count = deeplabv3_weight_count + embedding_weight_count
        avg_w = weight_sum / weight_count

        return avg_w, deeplabv3_avg_w, embedding_avg_w

    def print_config(self):
        print('Current config: embedding_dim = {}, num_shape_per_class = {}, class_ids = {}, xy_coordinates = {}'.format(
            self.embedding_dim, self.num_shape_per_class, self.class_ids, self.xy_coordinates))
    
    def show_results(self):
        self.check_and_create_dir()
        self.init_metrics_history()
        self.print_config()
        self.metrics_hist.plot()
        plt.show()

    def train(self):
        shapes_gen = shapes_generator(self.params_train_random)
        self.model.fit_generator(
            generator=shapes_gen, 
            steps_per_epoch=self.steps_per_epoch, 
            epochs=self.current_training_epoch+1, 
            initial_epoch=self.current_training_epoch,
            workers=self.workers,
            use_multiprocessing=False, 
            verbose=1)
        self.current_training_epoch += 1
    
    def evaluate(self):
        params = self.params_eval
        dataType = params.DATASET_TYPE
        dt_filename = 'shapes.json'
        result_list = self.get_embedding_result_list()
        dt_path = os.path.join(self.dt_dir, dt_filename)

        with open(dt_path, 'w') as outfile:
            json.dump(result_list, outfile)
        print('generated dt file at: {}'.format(dt_path))

        args = params_lib.Args()
        args.small                  = self.params_eval.SMALL
        args.class_ids              = self.class_ids
        args.num_shape_per_class    = self.num_shape_per_class
        args.dataType               = dataType
        args.dt_dir                 = self.dt_dir
        args.gt_dir                 = self.gt_json_dir
        args.maxProp                = int(1000)
        args.outputFile             = 'output'

        metrics = batchEval.main(args)
        return metrics

    def get_embedding_result_list(self):
        params = self.params_eval
        use_gt = self.use_gt
        result_list = []
        shapes.get_shapes.idx = 0
        num_sample = len(params.INDICES)
            
        for i in range(num_sample):
            if i%100 == 0:
                print('Evaluating {} out of {} {} examples'.format(i+1, num_sample, params.DATASET_TYPE))
            if use_gt:
                instances, image_info = single_eval(params)
            else:
                instances, image_info = single_eval(params, self.model)

            for instance in instances:
                encoded_mask = Mask.encode(np.asfortranarray(instance['mask']))
                encoded_mask = encoded_mask[0]
                # must convert bytes to string after encoding mask due to compatibility changes
                # for python 3
                encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")
                individual_result_dict = {
                    'segmentation': encoded_mask,
                    'score': instance['score'],
                    'image_id': str(image_info['image_id']),
                    'category_id': 1
                }
                result_list.append(individual_result_dict)

        return result_list
    
    def generate_dataset_json(self):
        gt_json_dir             = self.gt_json_dir
        params                  = self.params_val_non_random

        dataset_name            = params.DATASET_NAME
        dataset_type            = params.DATASET_TYPE
        small                   = params.SMALL
        num_shape_per_class     = params.NUM_SHAPE_PER_CLASS
        DR                      = params.DOWNSAMPLE_RESOLUTION
        image_dir               = params.IMAGE_DIR
        indices                 = params.INDICES
        side                    = params.SIDE
        class_ids               = params.CLASS_IDS

        if small:
            size_str = "small"
        else:
            size_str = "big"

        json_filename = os.path.join(gt_json_dir, dataset_name+"_"+dataset_type+"_"+size_str+"_"+str(num_shape_per_class)+"_"+str(class_ids)+".json")
        
        if not os.path.isfile(json_filename):
            print('generating gt json at: {}'.format(json_filename))
            name_dict = {
                1: 'circle',
                2: 'triangle',
                3: 'rectangle'
            }
            num_img = len(indices)
                
            json_dict = {}
            json_images = []
            json_annotations = []
            
            for i in range(num_img):
                if (i%100 == 0):
                    clear_output()
                    
                if (i%10 == 0):
                    print('Generating gt json data for image {}'.format(i))
                
                # images
                image_id = str(i)
                json_img_instance = {
                    'license': 1,
                    'filename': os.path.join(image_dir, image_id+".png"),
                    'height': side,
                    'width': side,
                    'id': image_id
                }
                json_images.append(json_img_instance)

                # annotations
                regions = []
                image_info = shapes.get_shapes(params)
                gt_instances = image_info['gt_instances']
                num_layer = len(np.unique(image_info['first_layer_mask']))
                for j in range(1, num_layer):
                    # read modal mask and resize
                    modal_mask = image_info['first_layer_mask']
                    modal_mask = (modal_mask==j)
                    class_mask = image_info['class_mask']
                    classnames = np.unique(class_mask[modal_mask==1])
                    assert len(classnames)==1
                    class_id = classnames[0]
                    
                    # read amodal mask and resize
                    amodal_mask = gt_instances[:, :, j:(j+1)]
                    amodal_mask = resize(amodal_mask, [DR, DR], order=0, mode='constant', preserve_range=True)
                    
                    # encode segmentation
                    encoded_mask = Mask.encode(np.asfortranarray(amodal_mask).astype(np.uint8))
                    encoded_mask = encoded_mask[0]
                    encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")

                    # calculate occlude rate, defined as the fraction of region area occluded

                    occlude_rate = (np.sum(amodal_mask) - np.sum(modal_mask))/np.sum(amodal_mask)

                    region = {
                        'segmentation': encoded_mask,
                        'name': name_dict[class_id],
                        'area': int(Mask.area([encoded_mask])[0]),
                        'isStuff': 0,
                        'occlude_rate': occlude_rate,
                        # according to coco amodal api, apparently the exact value of order does 
                        # not matter as long as it is greater than 0, which is reserved for 
                        # matching with the background
                        'order': j+1
                    }
                    regions.append(region)

                json_ann_instance = {
                    'image_id': image_id,
                    'id': i,
                    'regions': regions,
                    'size': len(regions)
                }
                json_annotations.append(json_ann_instance)

            json_dict = {
                'images': json_images,
                'annotations': json_annotations
            }

            with open(json_filename, 'w') as outfile:
                json.dump(json_dict, outfile)


def get_mrcnn_result_list(model, params):
    DR = params.DOWNSAMPLE_RESOLUTION
    shapes.get_shapes.idx = 0
    result_list = []
    set_type = params.DATASET_TYPE
    dataset = params.MRCNN_DATASET
    image_ids = dataset.image_ids
    num_samples = len(image_ids)
    for image_id in image_ids:
        if (image_id % 100 == 0):
            print('Evaluating {} out of {} {} examples'.format(image_id+1, num_samples, set_type))
        image = dataset.load_image(image_id)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        masks = r['masks'].astype(np.uint8)
        scores = r['scores']
        for i in range(len(scores)):
            mask = masks[:, :, i:i+1]
            mask = resize(mask, [DR, DR], order=0, mode='constant', preserve_range=True)
            mask = mask.astype(np.uint8)
            score = scores[i].astype(np.float64)
            encoded_mask = Mask.encode(np.asfortranarray(mask))
            encoded_mask = encoded_mask[0]
            # must convert bytes to string after encoding mask due to compatibility changes for python 3
            encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")
            individual_result_dict = {
                'segmentation': encoded_mask,
                'score': score,
                'image_id': str(image_id),
                'category_id': 1
            }
            result_list.append(individual_result_dict)
    return result_list 


def single_eval(params, model=None, display=False):

    class_num                       = params.CLASS_NUM
    embedding_dim                   = params.EMBEDDING_DIM
    use_gt                          = params.USE_GT
    n_layer_gt                      = params.N_LAYER_GT
    use_seg                         = params.USE_SEG
    ETH_mean_shift_threshold        = params.ETH_MEAN_SHIFT_THRESHOLD

    DR                              = params.DOWNSAMPLE_RESOLUTION
    small                           = params.SMALL

    instances = []

    if (use_gt == 1):
        image_info = get_shapes(params)
        mask_1_small = np.squeeze(image_info['mask_1'])
        mask_2_small = np.squeeze(image_info['mask_2'])
        mask_3_small = np.squeeze(image_info['mask_3'])
        mask_4_small = np.squeeze(image_info['mask_4'])
        mask_5_small = np.squeeze(image_info['mask_5'])
        mask_6_small = np.squeeze(image_info['mask_6'])
        mask_7_small = np.squeeze(image_info['mask_7'])
        layers = [mask_1_small, mask_2_small, mask_3_small, 
            mask_4_small, mask_5_small, mask_6_small, mask_7_small]
        # use gt masks for evaluation
        num_dt = int(np.max(mask_1_small))
        for i in range(num_dt):
            instance = {}
            temp = np.zeros((DR, DR))
            for j in range(n_layer_gt):
                temp[layers[j] == i+1] = 1
            instance['mask'] = np.expand_dims(temp, axis=-1).astype(np.uint8)
            instance['score'] = 1.0
            instances.append(instance)
    else:
        image_info, gt = get_single_image_and_gt(params)
        image = image_info['image']
        if small == False:
            original_size = image_info['original_size']
        image = image[None, :]
        if (len(image.shape) != 4):
            image = np.expand_dims(image, -1)
        if (image.shape[-1] != 3):
            image = np.concatenate((image, image, image), -1)

        # get predicted individual instance masks (final results)
        individual_masks_array = []

        # ground truth
        front_class_mask_gt             = gt[:, :, 0]
        back_class_mask_gt              = gt[:, :, 1]
        mask_gt                         = gt[:, :, 2]
        occ_mask_gt                     = gt[:, :, 3]

        combined_mask_gt = np.zeros((DR, 2*DR))
        combined_mask_gt[:, :DR] = mask_gt
        combined_mask_gt[:, DR:] = occ_mask_gt

        # predictions
        x = model.predict(image)

        front_class_mask_pred           = x[0, :, :, :class_num]
        back_class_mask_pred            = x[0, :, :, class_num:(2*class_num)]
        front_embedding_pred            = x[0, :, :, (2*class_num):(2*class_num + embedding_dim)]
        back_embedding_pred             = x[0, :, :, (2*class_num + embedding_dim):(2*class_num + 2*embedding_dim)]

        if use_seg == 0:
            # disable front and back classification results and assume 1 everywhere
            front_class_mask_int_pred = np.ones((DR, DR))
            back_class_mask_int_pred = np.ones((DR, DR))
        elif use_seg == 1:
            # use segmentation predicted by model
            front_class_mask_int_pred = np.argmax(front_class_mask_pred, axis=-1)
            back_class_mask_int_pred = np.argmax(back_class_mask_pred, axis=-1)
        elif use_seg == 2:
            # use gt segmentation for ablation study
            front_class_mask_int_pred = front_class_mask_gt
            back_class_mask_int_pred = back_class_mask_gt
            
        # Post-processing
        # separate the process for different non-background classes
        cluster_all_class = np.zeros((DR, DR*2))
        previous_highest_label = 0
        instance_to_class = []
        for j in range(class_num-1):
            mask_pred = np.zeros((DR, DR))
            occ_mask_pred = np.zeros((DR, DR))
            mask_pred[front_class_mask_int_pred == j+1] = 1
            occ_mask_pred[back_class_mask_int_pred == j+1] = 1
            combined_emb = np.zeros((DR, 2*DR, embedding_dim))
            combined_emb[:, :DR, :] = np.copy(front_embedding_pred)
            combined_emb[:, DR:, :] = np.copy(back_embedding_pred)
            combined_mask_pred = np.zeros((DR, 2*DR))
            combined_mask_pred[:, :DR] = mask_pred
            combined_mask_pred[:, DR:] = occ_mask_pred
            cluster = pp.ETH_mean_shift(
                combined_emb, combined_mask_pred, threshold=ETH_mean_shift_threshold)
            instance_to_class += [j+1] * np.max(cluster).astype(np.int)
            cluster[cluster != 0] += previous_highest_label
            filter_mask = combined_mask_pred > 0
            filter_template = np.zeros((DR, 2*DR))
            filter_template[filter_mask] = 1
            cluster = np.multiply(cluster, filter_template)
            cluster_all_class += cluster
            previous_highest_label = np.max(cluster_all_class)
        
        # process individual masks and put them in an array
        display_range = 10
        num_pixel_cutoff = 5
        num_dt = int(previous_highest_label)
        individual_masks_size_array = []
        for i in range(num_dt):
            instance = {}
            temp = np.zeros((DR, DR))
            temp[cluster_all_class[:, :DR] == i+1] = 1
            temp[cluster_all_class[:, DR:] == i+1] = 1
            mask_size = np.sum(temp)
            if (mask_size < num_pixel_cutoff):
                continue
            individual_masks_size_array.append(mask_size)
            individual_masks_array.append(temp)
            if small == 0:
                temp = resize(temp, original_size, order=0, mode='constant', preserve_range=True)
            instance['mask'] = np.expand_dims(temp, axis=-1).astype(np.uint8)
            class_idx_for_instance = instance_to_class[i]
            class_score_front = front_class_mask_pred[:, :, class_idx_for_instance]
            instance_mask_front = cluster_all_class[:, :DR] == i+1
            class_score_back = back_class_mask_pred[:, :, class_idx_for_instance]
            instance_mask_back = cluster_all_class[:, DR:] == i+1
            instance_score_front = np.multiply(class_score_front, instance_mask_front)
            instance_score_back = np.multiply(class_score_back, instance_mask_back)
            instance_score = instance_score_front + instance_score_back
            instance_score = np.sum(instance_score) / (np.sum(instance_mask_front) + np.sum(instance_mask_back))
            instance['score'] = instance_score
            instances.append(instance)
        # if there is no detection, add a dummy one so that evaluator does not crash
        if (len(instances) == 0):
            instance = {}
            if small == 0:
                fake_mask = np.ones(original_size, dtype=np.uint8)
                fake_mask = np.expand_dims(fake_mask, axis=-1)
            else:
                fake_mask = np.ones((DR, DR, 1), dtype=np.uint8)
            instance['mask'] = fake_mask
            instance['score'] = 0.99
            instances.append(instance)
    
        if display:
            imgId = image_info['image_id']
            image = np.squeeze(normalize(image))
            image = resize(image, (DR*2, DR*2))
            
            # pca on embedding for better visualization
            front_embedding_pred_flat = np.reshape(front_embedding_pred, (-1, embedding_dim))
            back_embedding_pred_flat = np.reshape(back_embedding_pred, (-1, embedding_dim))
            num_pixels = front_embedding_pred_flat.shape[0]
            total_embedding_pred_flat = np.zeros((2*num_pixels, embedding_dim))
            total_embedding_pred_flat[:num_pixels, :] = front_embedding_pred_flat
            total_embedding_pred_flat[num_pixels:, :] = back_embedding_pred_flat
            total_embedding_pred_flat = StandardScaler().fit_transform(total_embedding_pred_flat)
            pca = PCA(n_components=3)
            total_pc_flat = pca.fit_transform(total_embedding_pred_flat)
            front_pc_flat = total_pc_flat[:num_pixels, :]
            back_pc_flat = total_pc_flat[num_pixels:, :]
            front_pc = np.reshape(front_pc_flat, (DR, DR, 3))
            back_pc = np.reshape(back_pc_flat, (DR, DR, 3))
            norm_temp = np.zeros((DR, 2*DR, 3))
            norm_temp[:, :DR, :] = front_pc
            norm_temp[:, DR:, :] = back_pc
            norm_temp = normalize(norm_temp)
            front_pc = norm_temp[:, :DR, :]
            back_pc = norm_temp[:, DR:, :]
            
            # prepare predicted embeddings (front/back)
            front_show_mask = np.expand_dims(front_class_mask_int_pred > 0, axis=-1)
            back_show_mask = np.expand_dims(back_class_mask_int_pred > 0, axis=-1)
            front_embedding_masked = np.multiply(front_pc, front_show_mask)
            back_embedding_masked = np.multiply(back_pc, back_show_mask)
            img_emb_masked = np.zeros(shape=(2*DR, DR, 3))
            img_emb_masked[:DR, :, :] = front_embedding_masked
            img_emb_masked[DR:, :, :] = back_embedding_masked
            
            # show instance mask and predicted embeddings
            random_colors = np.random.random((int(np.max(cluster_all_class)), 3))
            all_instances = np.zeros((DR*2, DR, 3))
            slice_0 = np.zeros((DR*2, DR))
            slice_1 = np.zeros((DR*2, DR))
            slice_2 = np.zeros((DR*2, DR))
            cluster_all_class_vertical = np.zeros((DR*2, DR))
            cluster_all_class_vertical[:DR, :] = cluster_all_class[:, :DR]
            cluster_all_class_vertical[DR:, :] = cluster_all_class[:, DR:]

            for i in range(int(np.max(cluster_all_class_vertical))):
                slice_0[cluster_all_class_vertical == i] = random_colors[i, 0]
                slice_1[cluster_all_class_vertical == i] = random_colors[i, 1]
                slice_2[cluster_all_class_vertical == i] = random_colors[i, 2]
            all_instances[:, :, 0] = slice_0
            all_instances[:, :, 1] = slice_1
            all_instances[:, :, 2] = slice_2

            combined_mask_gt_color = np.zeros((DR, DR*2, 3))
            for i in np.unique(combined_mask_gt):
                combined_mask_gt_color[combined_mask_gt == i] = np.random.random((3))
            combined_mask_gt_color_vertical = np.zeros((DR*2, DR, 3))
            combined_mask_gt_color_vertical[:DR, :, :] = combined_mask_gt_color[:, :DR, :]
            combined_mask_gt_color_vertical[DR:, :, :] = combined_mask_gt_color[:, DR:, :]

            class_mask_int_pred = np.zeros((DR*2, DR))
            class_mask_int_pred[:DR, :] = front_class_mask_int_pred
            class_mask_int_pred[DR:, :] = back_class_mask_int_pred
            class_mask_int_pred_color = np.zeros((DR*2, DR, 3))
            class_mask_int_pred_color[:, :, 0] = class_mask_int_pred/3
            class_mask_int_pred_color[:, :, 1] = class_mask_int_pred/3
            class_mask_int_pred_color[:, :, 2] = class_mask_int_pred/3

            board = np.zeros((DR*2, DR*7, 3))
            board[:, :(DR*2), :] = image
            board[:DR, (DR*2):(DR*3), :] = front_pc
            board[DR:, (DR*2):(DR*3), :] = back_pc
            board[:, (DR*3):(DR*4), :] = class_mask_int_pred_color
            board[:, (DR*4):(DR*5), :] = img_emb_masked
            board[:, (DR*5):(DR*6), :] = all_instances
            board[:, (DR*6):(DR*7), :] = combined_mask_gt_color_vertical

            class_mask_gt_vertical = np.zeros((64*2, 64))
            class_mask_gt_vertical[:64, :] = front_class_mask_gt
            class_mask_gt_vertical[64:, :] = back_class_mask_gt

            plt.figure(figsize=(4*4, 4*2))
            plt.title("image id = {}".format(imgId))
            plt.imshow(board)

            # plt.figure(figsize=(6, 6))
            # plt.imshow(image)
            # plt.title('image')

            # plt.figure(figsize=(4, 8))
            # plt.imshow(board[:, (DR*2):(DR*3), :])
            # plt.title('embedding')

            # plt.figure(figsize=(4, 8))
            # plt.imshow(board[:, (DR*3):(DR*4), :])
            # plt.title('class mask')

            # plt.figure(figsize=(4, 8))
            # plt.imshow(board[:, (DR*4):(DR*5), :])
            # plt.title('masked embedding')

            # plt.figure(figsize=(4, 8))
            # plt.imshow(board[:, (DR*5):(DR*6), :])
            # plt.title('result')

            # plt.figure(figsize=(4, 8))
            # plt.imshow(board[:, (DR*6):(DR*7), :])
            # plt.title('gt')

            # plt.figure(figsize=(4, 8))
            # plt.imshow(class_mask_gt_vertical)
            # plt.title('class_mask_gt')

            # show up to 10 masks sorted by size
            individual_masks = np.zeros((DR, DR*display_range))
            mask_size_sort_idx = np.argsort(individual_masks_size_array)
            mask_size_sort_idx = mask_size_sort_idx[::-1]
            individual_masks_array = [individual_masks_array[idx] for idx in mask_size_sort_idx]
            for i in range(display_range):
                if (i < len(individual_masks_array)):
                    individual_masks[:, (DR*i):(DR*(i+1))] = individual_masks_array[i]
            #         plt.figure(figsize=(4, 4))
            #         plt.imshow(np.squeeze(individual_masks_array[i]))
            plt.figure(figsize=(4*display_range, 4))
            plt.title('Predicted individual instance masks')
            plt.imshow(individual_masks)

    if not display:
        return instances, image_info