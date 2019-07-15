import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from dataset import (get_cornell_grasp_ids, get_cornell_id_meta, cornell_grasp_id2realpath,
                            parse_pcl, pc2depth, parse_bbox, normalize_depth, parse_img)
from models import GraspModel
from utils import (center_crop, preprocess_input,feature2bboxwdeg)

def preprocess_raw(img_, depth_):
    dim = img_.shape[:2]
    img = np.copy(img_)
    depth = np.copy(depth_)
    d = normalize_depth(center_crop(depth, crop_size=cfg.crop_size))
    img = center_crop(img, crop_size=cfg.crop_size)
    img[...,-1] = d
    img = preprocess_input(img)
    return img, dim
def bbox_postprocess(bbox, dim):
    crop_b = cfg.crop_size//2
    center_x = dim[1]//2
    center_y = dim[0]//2
    left_c = max(0, center_x-crop_b)
    up_c = max(0, center_y-crop_b)
    bbox = bbox*cfg.crop_size
    bbox[...,0] += left_c
    bbox[...,1] += up_c
    return bbox

def inference_bbox(model, inputs, threshold=0.0):
    inp = torch.from_numpy(np.transpose(inputs, (0, 3, 1, 2))) # (b, h, w, c) -> (b, c, h, w)
    model.eval()
    with torch.no_grad():
        bboxes, degs, confs = feature2bboxwdeg(model(inp).detach().cpu().numpy(), threshold)
    return bboxes, degs, confs

def predict(model, rgbs, depths, threshold=0.0):
    model.eval()
    with torch.no_grad():
        rgds, dims = [], []
        for rgb, depth in zip(rgbs, depths):
            rgd, dim = preprocess_raw(rgb, depth)
            rgds += [rgd]
            dims += [dim]
        rgds = np.asarray(rgds,dtype=np.float32)
        bboxes, degs, confs = inference_bbox(model, rgds, threshold=threshold)
        for i in range(len(bboxes)):
            bboxes[i] = [ bbox_postprocess(bbox, dims[i]) for bbox in bboxes[i] ]
    return bboxes, degs, confs
