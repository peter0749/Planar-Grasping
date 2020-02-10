import os
import torch
import numpy as np
import random

SEED = 113 # For reproducability
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

cornell_dataset_root = './DATA/raw_data/fg'
cornell_dataset_meta_root = './DATA/processedData'
cornell_dataset_root = os.path.abspath(cornell_dataset_root)
cornell_dataset_meta_root = os.path.abspath(cornell_dataset_meta_root)

n_folds = 5 # 5 in paper
max_sample_bbox = 50 # 5 in paper
max_pad_bbox = 50 # unknown in paper
lambda_coord = 5   # Same as YOLOv1
lambda_rot = 3
hinge_margin = 1
iou_threshold = 0.4 # 0.25 in paper
deg_threshold = 15   # 30   in paper
grasp_threshold = 0.3 # Unknown in paper
iou_threshold_easy = 0.25
deg_threshold_easy = 30

n_orientations = 20
orientation = np.linspace(0, np.pi, n_orientations+1)[:-1]
orientation_base = orientation[1]
crop_range = 60
crop_size = 352 # 320 in paper
input_size = 352 # 224 in paper
grid_scale = 32 # 2^5
assert input_size%grid_scale==0
grid_size = input_size//grid_scale # (7?)

