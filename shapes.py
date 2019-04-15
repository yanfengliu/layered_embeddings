import numpy as np
from PIL import Image, ImageDraw
from scipy.special import binom
import cv2
import pickle
import os
from utils import consecutive_integer, totuple, add_xy
from keras.utils import to_categorical
from skimage.transform import resize


def bezier(points, num=200):
    bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


def get_random_shapes(params):
    num_shape_low   = params.NUM_SHAPE_LOW
    num_shape_high  = params.NUM_SHAPE_HIGH
    side            = params.SIDE

    num = np.random.randint(num_shape_low, num_shape_high)
    rad = 0.2
    edgy = 0.05
    image = np.zeros((side, side, 3), dtype=np.uint8) + 255
    for i in range(num):
        x_shift = np.random.randint(0, 0.6 * side)
        y_shift = np.random.randint(0, 0.6 * side)
        a = get_random_points(n=7, scale=0.3*side)
        x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
        points = np.zeros((1, len(x), 2))
        points[0, :, 0] = x + x_shift
        points[0, :, 1] = y + y_shift
        points = points.astype(np.int32)
        image = cv2.fillPoly(image, points, (255, 255, 255))
        for i in range(len(x)-1):
            pt1 = tuple(points[0, i, :])
            pt2 = tuple(points[0, i+1, :])
            image = cv2.line(image, pt1, pt2, (0, 0, 0), 10)
    return image


def get_new_shapes(params):

    # Convert the numpy array to an Image object.
    side                = params.SIDE
    num_shape_per_class = params.NUM_SHAPE_PER_CLASS
    class_ids           = params.CLASS_IDS

    int_to_shape = {
        1: "circle",
        2: "triangle",
        3: "rectangle"
    }
    shapes_info = []
    shape_list = []
    for class_id in class_ids:
        shape_list += [class_id] * num_shape_per_class

    np.random.shuffle(shape_list)

    for i in range(len(class_ids)*num_shape_per_class):
        shape_info = {}
        shape_info['side'] = side
        # randomly choose circle, triangle, or rectangle to draw
        shape_choice_int = shape_list[i]
        shape_choice_str = int_to_shape[shape_choice_int]
        shape_info['shape_choice_int'] = shape_choice_int
        shape_info['shape_choice_str'] = shape_choice_str

        # associate that instance label with the shape
        angle = np.random.randint(0, 360)
        theta = (np.pi / 180.0) * angle
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        x_shift = np.round((np.random.rand() * 0.8 + 0.1)*side)
        y_shift = np.round((np.random.rand() * 0.8 + 0.1)*side)
        shape_info['x_shift'] = x_shift
        shape_info['y_shift'] = y_shift
        offset = np.array([x_shift, y_shift])
        if (shape_choice_str == "circle" or shape_choice_str == "ellipse"):
            if shape_choice_str == "circle":
                radius = 0.25 * side
                x0 = x_shift
                y0 = y_shift
                x1 = x0 + radius
                y1 = y0 + radius
            elif shape_choice_str == "ellipse":
                x0 = x_shift
                y0 = y_shift
                ellipse_x = (np.random.random() * 0.3 + 0.1) * side
                ellipse_y = (np.random.random() * 0.3 + 0.1) * side
                x1 = x0 + ellipse_x
                y1 = y0 + ellipse_y
            shape_info['x1'] = x1
            shape_info['y1'] = y1
        else:
            if (shape_choice_str == "triangle"):
                L = 0.3 * side
                corners = [[0, 0], [L, 0], [0.5 * L, 0.866 * L], [0, 0]]
            elif (shape_choice_str == "star"):
                L = 0.2 * side
                a = L/(1+np.sqrt(3))
                corners = [[0, L], [L-a, L+a], [L, 2*L], [L+a, L+a], 
                    [2*L, L], [L+a, L-a], [L, 0], [L-a, L-a], [0, L]]
            elif (shape_choice_str == "rectangle"):
                width = 0.1 * side
                height = 0.8 * side
                corners = [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
            elif (shape_choice_str == "square"):
                width = 0.3 * side
                height = width
                corners = [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
            
            corners = np.dot(corners, R) + offset
            shape_info['corners'] = corners

        shapes_info.append(shape_info)

    return shapes_info


def draw_ellipse(draw, bbox, linewidth, image_or_mask, counter):

    if image_or_mask == 'image':
        for offset, fill in (linewidth/-2.0, 'black'), (linewidth/2.0, 'white'):
            left, top = [(value + offset) for value in bbox[:2]]
            right, bottom = [(value - offset) for value in bbox[2:]]
            draw.ellipse([left, top, right, bottom], fill=fill)
    elif image_or_mask == 'mask':
        offset, fill = (linewidth/-2.0, 'white')
        left, top = [(value + offset) for value in bbox[:2]]
        right, bottom = [(value - offset) for value in bbox[2:]]
        draw.ellipse([left, top, right, bottom], fill=counter)
        
    return draw


def get_rect(x, y, width, height, angle):
    """
    Utility function to create rectangles using PIL.ImageDraw
    """

    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def get_n_layer_gt_from_shapes(params, shapes_info):
    full_gt         = params.FULL_GT
    DR              = params.DOWNSAMPLE_RESOLUTION
    side            = params.SIDE

    # Image init
    img             = Image.new(mode='RGB', size=(side, side), color=(255, 255, 255))
    mask            = Image.new(mode='I', size=(side, side), color=0)
    occ_mask        = Image.new(mode='I', size=(side, side), color=0)
    class_mask      = Image.new(mode='I', size=(side, side), color=0)

    # Initialize Draw instance for image, mask, class mask, and occ_mask
    draw_img        = ImageDraw.Draw(img)
    draw_mask       = ImageDraw.Draw(mask)
    draw_occ_mask   = ImageDraw.Draw(occ_mask)
    draw_class_mask = ImageDraw.Draw(class_mask)

    draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask
    }

    # Draw background mask
    rect = get_rect(x=0, y=0, width=side, height=side, angle=0)
    draw_occ_mask.polygon([tuple(p) for p in rect], fill=1, outline=1)
    occ_mask_buffer = []
    occ_mask_buffer.append(occ_mask)

    counter = 1
    num = len(shapes_info)
    instance_to_class_temp = np.zeros(shape=(num+1))
    for i in range(num):
        shape_info = shapes_info[i]
        occ_mask = Image.new(mode='I', size=(side, side), color=0)
        draw_occ_mask = ImageDraw.Draw(occ_mask)
        draws['draw_occ_mask'] = draw_occ_mask
        draws = draw_shapes(shape_info, draws, counter)
        occ_mask_buffer.append(occ_mask)
        counter = counter + 1
        instance_to_class_temp[i+1] = shape_info['shape_choice_int']

    # Convert the Image data to a numpy array.
    image = np.asarray(img)
    image = np.copy(image)
    image = image / 255.0

    # Generate instance mask
    mask = np.asarray(mask)
    mask = np.copy(mask)
    mask = np.expand_dims(mask, axis=2)

    # Convert mask to consecutive integer values starting from 0
    mask, change_log = consecutive_integer(mask)

    # Generate initial occlusion mask by adding all occlusion masks together
    num_layer = len(np.unique(mask))
    occ_mask_img_buffer = np.zeros((side, side, num_layer))
    for i in range(num_layer):
        gt_instance = np.asarray(occ_mask_buffer[int(change_log[i])])
        occ_mask_img_buffer[:, :, i] = gt_instance
    stacking_mask = np.sum(occ_mask_img_buffer, axis=2)

    # Generate tri-state mask (number of objects at the corresponding pixel = 0, 1, or 2)
    num_stack_mask = np.copy(stacking_mask)
    for i in range(8):
        num_stack_mask[stacking_mask == i+1] = i

    # Generate occlusion mask that indicates the index of the occluded object
    # (used for finding the target embedding of the occluded object)

    def get_n_th_mask(n):
        mask_n = np.zeros((side, side))
        update_idx = (num_stack_mask >= n)
        obj_idx_array = occ_mask_img_buffer[update_idx]
        obj_idx_pool = np.linspace(0, num_layer-1, num_layer)
        obj_idx = np.multiply(obj_idx_pool, (obj_idx_array).astype(int))
        num_update_pixels = obj_idx.shape[0]
        obj_unique_idx = np.zeros((num_update_pixels, 1))
        for i in range(num_update_pixels):
            obj_unique_idx[i] = np.unique(obj_idx[i, :])[-n]
        mask_n[update_idx] = np.squeeze(obj_unique_idx)
        mask_n = np.expand_dims(mask_n, 2)
        return mask_n

    mask_2 = get_n_th_mask(2)
    mask_3 = get_n_th_mask(3)
    mask_4 = get_n_th_mask(4)
    mask_5 = get_n_th_mask(5)
    mask_6 = get_n_th_mask(6)
    mask_7 = get_n_th_mask(7)

    mask                = resize(mask, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask, change_log    = consecutive_integer(mask)

    mask_2_temp         = resize(mask_2, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask_3_temp         = resize(mask_3, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask_4_temp         = resize(mask_4, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask_5_temp         = resize(mask_5, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask_6_temp         = resize(mask_6, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask_7_temp         = resize(mask_7, [DR, DR], order=0, mode='constant', preserve_range=True)

    mask_2              = np.zeros(shape=[DR, DR, 1])
    mask_3              = np.zeros(shape=[DR, DR, 1])
    mask_4              = np.zeros(shape=[DR, DR, 1])
    mask_5              = np.zeros(shape=[DR, DR, 1])
    mask_6              = np.zeros(shape=[DR, DR, 1])
    mask_7              = np.zeros(shape=[DR, DR, 1])

    for i in range(len(change_log)):
        mask_2[mask_2_temp == change_log[i]] = i
        mask_3[mask_3_temp == change_log[i]] = i
        mask_4[mask_4_temp == change_log[i]] = i
        mask_5[mask_5_temp == change_log[i]] = i
        mask_6[mask_6_temp == change_log[i]] = i
        mask_7[mask_7_temp == change_log[i]] = i

    image_info = {
        'image':            image,
        'original_size':    [side, side],
        'mask_1':           mask,
        'mask_2':           mask_2,
        'mask_3':           mask_3,
        'mask_4':           mask_4,
        'mask_5':           mask_5,
        'mask_6':           mask_6,
        'mask_7':           mask_7
    }
    if full_gt:
        num_layer = len(np.unique(mask))
        gt_instances = np.zeros((DR, DR, num_layer))
        for i in range(1, num_layer):
            gt_instance = occ_mask_img_buffer[:, :, int(change_log[i])]
            gt_instance = resize(gt_instance, [DR, DR], order=0, mode='constant', preserve_range=True)
            gt_instances[:, :, i] = gt_instance
        image_info['gt_instances'] = gt_instances

    return image_info


def get_image_from_shapes(params, shapes_info):
    full_gt         = params.FULL_GT
    DR              = params.DOWNSAMPLE_RESOLUTION
    side            = params.SIDE

    # Image init
    img             = Image.new(mode='RGB', size=(side, side), color=(255, 255, 255))
    mask            = Image.new(mode='I', size=(side, side), color=0)
    class_mask      = Image.new(mode='I', size=(side, side), color=0)
    occ_mask        = Image.new(mode='I', size=(side, side), color=0)

    # Initialize Draw instance for image, mask, class mask, and occ_mask
    draw_img        = ImageDraw.Draw(img)
    draw_mask       = ImageDraw.Draw(mask)
    draw_class_mask = ImageDraw.Draw(class_mask)
    draw_occ_mask   = ImageDraw.Draw(occ_mask)

    draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask
    }

    # Draw background mask
    rect = get_rect(x=0, y=0, width=side, height=side, angle=0)
    draw_occ_mask.polygon([tuple(p) for p in rect], fill=1, outline=1)
    occ_mask_buffer = []
    occ_mask_buffer.append(occ_mask)

    counter = 1
    num = len(shapes_info)
    instance_to_class_temp = np.zeros(shape=(num+1))
    for i in range(num):
        shape_info = shapes_info[i]
        occ_mask = Image.new(mode='I', size=(side, side), color=0)
        draw_occ_mask = ImageDraw.Draw(occ_mask)
        draws['draw_occ_mask'] = draw_occ_mask
        draws = draw_shapes(shape_info, draws, counter)
        occ_mask_buffer.append(occ_mask)
        counter = counter + 1
        instance_to_class_temp[i+1] = shape_info['shape_choice_int']

    # Convert the Image data to a numpy array.
    image = np.asarray(img)
    image = np.copy(image)
    image = image / 255.0

    # Generate instance mask
    mask = np.asarray(mask)
    mask = np.copy(mask)
    mask = np.expand_dims(mask, axis=2)

    # Generate class mask
    class_mask = np.asarray(class_mask)
    class_mask = np.copy(class_mask)
    class_mask = np.expand_dims(class_mask, axis=2)

    # Convert mask to consecutive integer values starting from 0
    mask, change_log = consecutive_integer(mask)
    instance_to_class = np.zeros(change_log.shape)
    for i in range(len(change_log)):
        instance_to_class[i] = instance_to_class_temp[int(change_log[i])]

    # Generate initial occlusion mask by adding all occlusion masks together
    num_layer = len(np.unique(mask))
    occ_mask_img_buffer = np.zeros((side, side, num_layer))
    for i in range(num_layer):
        gt_instance = np.asarray(occ_mask_buffer[int(change_log[i])])
        occ_mask_img_buffer[:, :, i] = gt_instance
    stacking_mask = np.sum(occ_mask_img_buffer, axis=2)

    # Generate tri-state mask (number of objects at the corresponding pixel = 0, 1, or 2)
    tri_state_mask = np.copy(stacking_mask)
    tri_state_mask[stacking_mask == 1] = 0
    tri_state_mask[stacking_mask == 2] = 1
    tri_state_mask[stacking_mask >= 3] = 2

    # Generate occlusion mask that indicates the index of the occluded object
    # (used for finding the target embedding of the occluded object)
    occ_mask = np.zeros((side, side))
    update_idx = (tri_state_mask == 2)
    obj_idx_array = occ_mask_img_buffer[update_idx]
    obj_idx_pool = np.linspace(0, num_layer-1, num_layer)
    obj_idx = np.multiply(obj_idx_pool, (obj_idx_array).astype(int))
    num_update_pixels = obj_idx.shape[0]
    obj_unique_idx = np.zeros((num_update_pixels, 1))
    for i in range(num_update_pixels):
        obj_unique_idx[i] = np.unique(obj_idx[i, :])[-2]
    occ_mask[update_idx] = np.squeeze(obj_unique_idx)

    occ_mask = np.expand_dims(occ_mask, 2)

    mask             = resize(mask, [DR, DR], order=0, mode='constant', preserve_range=True)
    mask, change_log = consecutive_integer(mask)
    class_mask      = resize(class_mask, [DR, DR], order=0, mode='constant', preserve_range=True)
    occ_mask_temp   = resize(occ_mask, [DR, DR], order=0, mode='constant', preserve_range=True)
    occ_mask = np.zeros(shape=occ_mask_temp.shape)

    instance_to_class_final = np.zeros(change_log.shape)

    for i in range(len(change_log)):
        occ_mask[occ_mask_temp == change_log[i]] = i
        instance_to_class_final[i] = instance_to_class[int(change_log[i])]

    back_class_mask = np.zeros(shape=mask.shape)
    for i in range(len(change_log)):
        back_class_mask[occ_mask == i] = instance_to_class_final[i]

    image_info = {
        'image':            image,
        'original_size':    [side, side],
        'first_layer_mask': mask,
        'occ_mask':         occ_mask,
        'class_mask':       class_mask,
        'back_class_mask':  back_class_mask
    }
    if full_gt:
        num_layer = len(np.unique(mask))
        gt_instances = np.zeros((DR, DR, num_layer))
        for i in range(1, num_layer):
            gt_instance = occ_mask_img_buffer[:, :, int(change_log[i])]
            gt_instance = resize(gt_instance, [DR, DR], order=0, mode='constant', preserve_range=True)
            gt_instances[:, :, i] = gt_instance
        image_info['gt_instances'] = gt_instances

    return image_info


def draw_shapes(shape_info, draws, counter):
    side                = shape_info['side']
    x_shift             = shape_info['x_shift']
    y_shift             = shape_info['y_shift']
    shape_choice_str    = shape_info['shape_choice_str']
    shape_choice_int    = shape_info['shape_choice_int']

    draw_img            = draws['draw_img']
    draw_mask           = draws['draw_mask']
    draw_occ_mask       = draws['draw_occ_mask']
    draw_class_mask     = draws['draw_class_mask']

    line_width = int(0.01 * side)

    if (shape_choice_str == "circle" or shape_choice_str == "ellipse"):
        x0 = x_shift
        y0 = y_shift
        x1 = shape_info['x1']
        y1 = shape_info['y1']
        
        draw_ellipse(draw=draw_img, bbox=[x0, y0, x1, y1], linewidth=line_width, 
                    image_or_mask='image', counter=counter)
        draw_ellipse(draw=draw_mask, bbox=[x0, y0, x1, y1], linewidth=line_width, 
                    image_or_mask='mask', counter=counter)
        draw_ellipse(draw=draw_class_mask, bbox=[x0, y0, x1, y1], linewidth=line_width, 
                    image_or_mask='mask', counter=int(shape_choice_int))
        draw_ellipse(draw=draw_occ_mask, bbox=[x0, y0, x1, y1], linewidth=line_width, 
                    image_or_mask='mask', counter=1)
    else:
        corners = shape_info['corners']
        # get tuple version of the points
        shape_tuple = totuple(corners)
        # draw image
        draw_img.polygon(xy=shape_tuple, fill=(255, 255, 255), outline=0)
        # draw line around polygon to adjust line width since polygon doesn't support it
        draw_img.line(xy=shape_tuple, fill=(0, 0, 0), width=line_width)
        # draw ellipses around the corner to make lines look nice
        for point in shape_tuple:
            draw_img.ellipse((point[0] - 0.5*line_width, 
                            point[1] - 0.5*line_width, 
                            point[0] + 0.5*line_width, 
                            point[1] + 0.5*line_width), 
                            fill=(0, 0, 0))
        # draw instance mask
        draw_mask.polygon(xy=shape_tuple, fill=counter, outline=counter)
        # draw class mask
        draw_class_mask.polygon(xy=shape_tuple, fill=int(shape_choice_int), outline=int(shape_choice_int))
        # draw occlusion masks, one per layer; number of layers = num
        draw_occ_mask.polygon(shape_tuple, fill=1, outline=1)
    
    new_draws = {
        'draw_img': draw_img,
        'draw_mask': draw_mask,
        'draw_class_mask': draw_class_mask
    }
    
    return  new_draws


def get_shapes(params):
    sample_cache_dir    = params.SAMPLE_CACHE_DIR
    set_type            = params.DATASET_TYPE
    num_shape_per_class = params.NUM_SHAPE_PER_CLASS
    small               = params.SMALL
    class_ids           = params.CLASS_IDS
    XY                  = params.XY
    use_gt              = params.USE_GT

    if small:
        size_str = "small"
    else:
        size_str = "big"

    if set_type == 'train':
        shapes_info = get_new_shapes(params)
        image_info = get_image_from_shapes(params, shapes_info)
        image_info['image_id'] = 'random'
    else:
        if (params.RANDOM):
            img_id = int(np.random.random()*len(params.INDICES))
        else:
            img_id = get_shapes.idx
            get_shapes.idx += 1
        
        specific_dir = os.path.join(sample_cache_dir, set_type, str(num_shape_per_class), str(class_ids), str(XY), size_str, str(use_gt))
        if not os.path.isdir(specific_dir):
            os.makedirs(specific_dir)
        sample_cache_file_path = os.path.join(specific_dir, str(img_id)+'.pickle')
    
        if (os.path.isfile(sample_cache_file_path)):
            with open(sample_cache_file_path, 'rb') as handle:
                image_info = pickle.load(handle)
        else:
            shapes_info = get_new_shapes(params)
            if use_gt:
                image_info = get_n_layer_gt_from_shapes(params, shapes_info)
            else:
                image_info = get_image_from_shapes(params, shapes_info)

            with open(sample_cache_file_path, 'wb') as handle:
                pickle.dump(image_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        image_info['image_id'] = img_id
    
    if XY:
        image_info['image'] = add_xy(image_info['image'])

    return image_info