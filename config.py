import os
import torch
import numpy as np
import random

SEED = 29413 # For reproducability
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

cornell_dataset_root = './DATA/raw_data/fg'
cornell_dataset_meta_root = './DATA/processedData'
cornell_dataset_root = os.path.abspath(cornell_dataset_root)
cornell_dataset_meta_root = os.path.abspath(cornell_dataset_meta_root)

n_folds = 5
max_sample_bbox = 5
max_pad_bbox = 10
lambda_coord = 5   # Same as YOLOv1
lambda_rot = 7
hinge_margin = 1
iou_threshold = 0.5 # 0.25 in paper
deg_threshold = 15  # 30   in paper

crop_range = 40
crop_size = 320
input_size = 224
grid_scale = 32 # 2^5
assert input_size%grid_scale==0
grid_size = input_size//grid_scale # (7?)

