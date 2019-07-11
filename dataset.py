import os
import sys
import random
import re
import numpy as np
import numba as nb
import cv2
import glob
import torch
from torch.utils.data import Dataset
import utils
import config as cfg

@nb.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
      result = np.empty(arr.shape[1])
      for i in range(len(result)):
        result[i] = func1d(arr[:, i])
    else:
      result = np.empty(arr.shape[0])
      for i in range(len(result)):
        result[i] = func1d(arr[i, :])
    return result

@nb.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@nb.jit(nopython=True)
def normalize_depth(depth):
    depth = depth-depth.min()
    depth = depth/depth.max()*255
    return depth

def get_cornell_grasp_ids():
    prefix = cfg.cornell_dataset_root
    ids = []
    folds = []
    for img_name in glob.glob(prefix+'/*r.png'):
        id_ = os.path.splitext(os.path.split(img_name)[-1])[0][3:-1]
        ids += [id_]
    return ids

def split_cornell_grasp_ids_nfold_by_img(ids):
    random.shuffle(ids)
    ids_set = set(ids)
    folds = []
    step = len(ids)//cfg.n_folds
    for i in range(0, cfg.n_folds):
        b = step*i
        test = ids[b:b+step]
        train = list( ids_set - set(test)  )
        folds.append((train, test))
    return folds

def split_cornell_grasp_ids_nfold_by_obj(ids, obj2id):
    ids_set = set(ids)
    obj_list = list(obj2id)
    random.shuffle(obj_list)
    obj_set  = set(obj_list)
    folds = []
    step = len(obj_list)//cfg.n_folds
    for i in range(0, cfg.n_folds):
        b = step*i
        test = []
        for x in obj_list[b:b+step]:
            test += list(set(obj2id[x])&ids_set)
        train = []
        for x in list( obj_set - set(test)  ):
            train += list(set(obj2id[x])&ids_set)
        folds.append((train, test))
    return folds

def cornell_grasp_id2realpath(id_):
    prefix = cfg.cornell_dataset_root
    img = prefix+("/pcd{:s}r.png".format(id_))
    pcl = prefix+("/pcd{:s}.txt".format(id_))
    pcl_npy = prefix+("/pcd{:s}.npz".format(id_))
    pos = prefix+("/pcd{:s}cpos.txt".format(id_))
    pos_npy = prefix+("/pcd{:s}cpos.npy".format(id_))
    neg = prefix+("/pcd{:s}cneg.txt".format(id_))
    neg_npy = prefix+("/pcd{:s}cneg.npy".format(id_))
    return img, pcl, pos, neg, pcl_npy, pos_npy, neg_npy

def parse_pcl(path):
    pts = []
    indices = []
    n_lines = 0
    with open(path, mode='r') as fp:
        for line in fp.readlines():
            line_list = line.split()
            if bool(re.match("POINTS", line_list[0], re.I)):
                n_lines = int(line_list[1])
            if len(line_list)==5:
                floats = list(map(float, line_list))[:3]
                if len(floats)!=3:
                    continue
                index = int(line_list[4])
                pts += [ floats ]
                indices += [ index  ]
    pts = np.array(pts, dtype=np.float32)
    pts[pts!=pts] = 0
    indices = np.array(indices, dtype=np.int32)
    return pts, indices

def fetch_pcl(path):
    z = np.load(path)
    return z['pts'], z['indices']

def fetch_bbox(path):
    return np.load(path)

def parse_bbox(path):
    bboxes = []
    bbox = []
    with open(path, mode='r') as fp:
        for line in fp.readlines():
            xy = list(map(float, line.split()))[:2]
            bbox += [xy]
            if len(bbox)==4:
                bboxes += [bbox]
                bbox = []
    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes

def parse_img(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)[...,::-1] # RGB format

def get_cornell_id_meta():
    prefix = cfg.cornell_dataset_meta_root
    class_info_path = prefix+"/z_uniq.txt"
    meta = {}
    class2id = {}
    obj_id2id = {}
    with open(class_info_path, mode="r") as cls_fp:
        cls_fp_lines = cls_fp.readlines()
        for line_cls in cls_fp_lines:
            id_, obj_id, class_str = line_cls.split()[:3]
            id_ = id_.strip()
            obj_id = int(obj_id)
            class_str = class_str.strip()
            meta[id_] = (obj_id, class_str)
            if class_str in class2id:
                class2id[class_str].append(id_)
            else:
                class2id[class_str] = [id_]
            if obj_id in obj_id2id:
                obj_id2id[obj_id].append(id_)
            else:
                obj_id2id[obj_id] = [id_]
    return meta, class2id, obj_id2id

@nb.jit(nopython=True)
def pc2depth(pts, index, size=(480,640)):
    depth = np.full(size, pts.min(), dtype=np.float32)
    depth_view = depth.ravel()
    depth_view[index] = pts[...,2] # only depth
    return depth



@nb.jit(nopython=True)
def paramize_bbox(x):
    # shape of x: (4, 2); range: 0~1
    # TL, TR, DR, DL
    x = x * cfg.grid_size # range: 0~7 (?)
    xy = np_mean(x, 0) # center
    grid_idx_x = int(np.floor(xy[0])) # integer
    grid_idx_y = int(np.floor(xy[1])) # integer
    offset_x = xy[0] - grid_idx_x # decimal
    offset_y = xy[1] - grid_idx_y # decimal
    h  = np.linalg.norm(x[0]-x[1]) # related to grid scale
    w  = np.linalg.norm(x[1]-x[2])
    costha = (x[1,0]-x[0,0]) / (h+1e-8)  # (x[1].x - x[0].x) / norm(x[1]-x[0])
    sintha = -(x[1,1]-x[0,1]) / (h+1e-8) # (x[1].y - x[0].y) / norm(x[1]-x[0]) coord in img is upside down
    costha2 = 2*costha*costha-1  # Double-Angle formula
    sintha2 = 2*costha*sintha
    return grid_idx_x, grid_idx_y, offset_x, offset_y, w, h, costha2, sintha2

### Build in progress... ###
class CornellGraspDataset(Dataset):
    def __init__(self, split_mode='img', input_mode='RGD', return_box=False): # OR RGBD
        self.img_ids = get_cornell_grasp_ids()
        self.meta, self.class2id, self.obj2id = get_cornell_id_meta()
        if split_mode=='img':
            self.folds = split_cornell_grasp_ids_nfold_by_img(self.img_ids)
        else:
            self.folds = split_cornell_grasp_ids_nfold_by_obj(self.img_ids, self.class2id)
        self.current_state = 'train'
        self.current_fold = 0
        self.set_current_state('train')
        self.set_current_fold(0)
        self.input_mode = input_mode
        self.aug_spatial, self.aug_color = utils.get_imgaug()
        self.return_box = return_box

    def set_current_state(self, mode):
        self.current_state = 'train' if mode=='train' else 'test'
    def get_fold_number(self):
        return len(self.folds)
    def set_current_fold(self, f):
        self.current_fold = f
        self.init_current_fold()
    def get_current_fold(self):
        return self.current_fold
    def init_current_fold(self):
        self.train_id, self.test_id = self.folds[self.current_fold]
        self.current_state_fold_id = self.train_id if self.current_state=='train' else self.test_id
    def __getitem__(self, index):
        #return torch.zeros((3,224,224)), torch.zeros((7,7,7)), torch.zeros((10,4,2))
        img_p, _, __, ___ , pcl_p, pos_p, neg_p = cornell_grasp_id2realpath(self.current_state_fold_id[index])
        img = parse_img(img_p)
        pts, pts_idx = fetch_pcl(pcl_p)
        depth = normalize_depth(pc2depth(pts, pts_idx, size=img.shape[:2]))
        boxes = np.load(pos_p)
        # negative samples not used
        if self.current_state=='train' and len(boxes)>cfg.max_sample_bbox:
            boxes = boxes[np.random.choice(len(boxes), cfg.max_sample_bbox, replace=False)]

        crop_size=cfg.crop_size
        if self.current_state=='train':
            crop_size += random.randint(-cfg.crop_range, cfg.crop_range)
        img, boxes = utils.center_crop(img, boxes, crop_size=crop_size)
        depth = utils.center_crop(depth, crop_size=crop_size)

        if self.current_state=='train':
            seq_spatial = self.aug_spatial.to_deterministic()
            img, boxes = utils.aug_data(seq_spatial, img, boxes)
            depth = utils.aug_data(seq_spatial, depth)
            img = self.aug_color(image=img)

        ori_img_dim = img.shape[:2]
        boxes[...,0] /= ori_img_dim[1] # x : col
        boxes[...,1] /= ori_img_dim[0] # y : row
        model_input = img
        if self.input_mode=='RGD':
            model_input[...,-1] = depth
        else:
            model_input = np.append(model_input, depth[...,np.newaxis], axis=-1)
        model_input = utils.preprocess_input(model_input)
        label = np.zeros((cfg.grid_size,cfg.grid_size,7), dtype=np.float32) # (confidence,x,y,w,h,cos,sin)
        for box in boxes:
            # grid_idx_x, grid_idx_y, offset_x, offset_y, w, h, costha2, sintha2
            box = box.clip(1e-6, 1-1e-6)
            box[box!=box] = 0
            j, i, offx, offy, w, h, costha2, sintha2 = paramize_bbox(box)
            i = np.clip(i, 0, cfg.grid_size)
            j = np.clip(j, 0, cfg.grid_size)
            label[i,j,:] = 1.0, offx, offy, w, h, costha2, sintha2 # Degration of YOLOv1/v2 (lack of anchor boxes)
        model_input[model_input!=model_input] = 0 # get rid of nans
        label[label!=label] = 0
        model_input = torch.from_numpy( np.transpose(model_input, (2,0,1)).astype(np.float32)  ) # (h,w,c) -> (c,h,w)
        label = torch.from_numpy( np.transpose(label, (2,0,1)).astype(np.float32)  ) # (h,w,c) -> (c,h,w)
        if self.return_box and len(boxes)<cfg.max_pad_bbox:
            boxes = np.pad(boxes, ((0,cfg.max_pad_bbox-len(boxes)), (0,0), (0,0)), mode='constant', constant_values=0)
        if self.return_box and len(boxes)>cfg.max_pad_bbox:
            boxes = boxes[np.random.choice(len(boxes), cfg.max_pad_bbox, replace=False)]
        if self.return_box:
            assert boxes.shape==(cfg.max_pad_bbox, 4, 2)
            boxes = torch.from_numpy(boxes)

        return (model_input, label, boxes) if self.return_box else (model_input, label)
    def __len__(self):
        return len(self.current_state_fold_id)

if __name__=='__main__':
    '''
    ids = get_cornell_grasp_ids()
    meta, class2id, obj2id = get_cornell_id_meta()
    # Test id -> data
    for id_ in np.random.choice(ids, 3):
        img_p, pcl_p, pos_p, neg_p = cornell_grasp_id2realpath(id_)
        img = parse_img(img_p)
        pts, idx = parse_pcl(pcl_p)
        depth = pc2depth(pts, idx, size=img.shape[:2])
        boxes = parse_bbox(pos_p)
        print(depth.shape)
        print(boxes.shape)

    # Test kfold (image wise)
    folds = split_cornell_grasp_ids_nfold_by_img(ids)
    for (train,test) in folds:
        print(len(train), len(test))
        if False:
            for t in [train,test]:
                for id_ in t:
                    img_p, pcl_p, pos_p, neg_p = cornell_grasp_id2realpath(id_)
                    print(parse_pcl(pcl_p).shape)
    # Test kfold (obj wise)
    folds = split_cornell_grasp_ids_nfold_by_obj(ids, obj2id)
    for (train,test) in folds:
        print(len(train), len(test))
        if False:
            for t in [train,test]:
                for id_ in t:
                    img_p, pcl_p, pos_p, neg_p = cornell_grasp_id2realpath(id_)
                    print(parse_pcl(pcl_p).shape)
    '''
    # Test Dataset
    dataset = CornellGraspDataset(return_box=True)
    for i in range(3):
        dataset[i]
        print(i+1, 3)
    '''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    i = 0
    for (a,b,c) in dataloader:
        i+=1
        if i==3:
            break
    '''
