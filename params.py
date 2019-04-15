import numpy as np
import os

class Params(object):

    def __init__(self, experiment):
        self.EMBEDDING_DIM              = experiment.embedding_dim
        self.XY                         = experiment.xy_coordinates
        self.NUM_SHAPE_PER_CLASS        = experiment.num_shape_per_class
        self.BATCH_SIZE                 = experiment.batch_size
        self.SAMPLE_CACHE_DIR           = experiment.sample_cache_dir
        self.CLASS_NUM                  = experiment.class_num
        self.NUM                        = experiment.val_sample_num
        self.CLASS_IDS                  = experiment.class_ids
        self.DOWNSAMPLE_FACTOR          = experiment.downsample_factor
        self.NUM_FILTER                 = experiment.num_filter
        self.ETH_MEAN_SHIFT_THRESHOLD   = experiment.eth_mean_shift_threshold
        self.DELTA_VAR                  = experiment.delta_var
        self.DELTA_D                    = experiment.delta_d
        self.SIDE                       = experiment.side
        self.BACKBONE                   = experiment.backbone
        self.VERBOSE                    = experiment.verbose
        self.USE_GT                     = experiment.use_gt
        self.SEMANTIC_ONLY              = experiment.semantic_only
        self.DATA_AUG                   = experiment.data_aug
        self.IMAGE_DIR                  = experiment.image_dir
        self.DATASET_NAME               = experiment.dataset_name
        self.N_LAYER_GT                 = experiment.n_layer_gt
        self.USE_SEG                    = experiment.use_seg
        
        self.DOWNSAMPLE_RESOLUTION = int(self.SIDE / self.DOWNSAMPLE_FACTOR)
        self.INDICES = np.linspace(0, self.NUM-1, self.NUM).astype(np.int)
    
    def display_values(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
    
class ParamsTrainRandom(Params):        
    FULL_GT = False
    SMALL = True
    RANDOM = True
    DATASET_TYPE = 'train'

class ParamsValRandom(Params):
    FULL_GT = True
    SMALL = True
    RANDOM = True
    DATASET_TYPE = 'val'

class ParamsValNonrandom(Params):
    FULL_GT = True
    SMALL = True
    RANDOM = False
    DATASET_TYPE = 'val'

class ParamsTestNonrandom(Params):
    FULL_GT = True
    SMALL = True
    RANDOM = False
    DATASET_TYPE = 'test'

class Args(object):
    pass