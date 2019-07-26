import os
import re
import tempfile
import selectivesearch
import torch
import matplotlib.pyplot as plt
import numpy as np
from pydarknet import Image, Detector
from . import config as cfg
from .dataset import (get_cornell_grasp_ids, get_cornell_id_meta, cornell_grasp_id2realpath,
                            parse_pcl, pc2depth, parse_bbox, normalize_depth, parse_img)
from .models import GraspModel
from .utils import (center_crop, crop_image, preprocess_input,feature2bboxwdeg)

def fix_yolo_data_path(data):
    for n, line in enumerate(data):
        i = re.search('( |=)data/', line)
        if i: # find and replace
            i = i.span()[0]
            data[n] = (line[:i+1] + os.path.join(os.path.split(__file__)[0], line[i+1:])).strip()
    return data

class GraspDetector(object):
    def __init__(self, backbone: str = 'vgg19_bn', weight_path: str = os.path.join(os.path.split(__file__)[0],"weights","grasp_model.pth"), f16: bool = False, cuda: bool = True,
            yolo_cfg: str = os.path.join(os.path.split(__file__)[0],"cfg","yolo9000.cfg"), yolo_weight: str = os.path.join(os.path.split(__file__)[0],"weights","yolo9000.weights"), yolo_data: str = os.path.join(os.path.split(__file__)[0],"cfg","combine9k.data")
            ,**kwargs):
        self.model = GraspModel(backbone=backbone, with_fc=False)
        if cuda:
            self.model = self.model.cuda()
        state_dict = torch.load(weight_path) if cuda else torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.half() if f16 else self.model.float()
        self.model.eval()
        self.yolo = None
        ### FIX RELATIVE PATH ISSUE ###
        with open(yolo_data, 'r') as fp:
            yolo_data_content = fp.readlines()
        with open(yolo_cfg, 'r') as fp:
            yolo_cfg_content = fp.readlines()
        with tempfile.NamedTemporaryFile() as tmp_yolo_cfg:
            for line in fix_yolo_data_path(yolo_cfg_content):
                tmp_yolo_cfg.write(bytes(line+'\n', encoding="utf-8"))
            tmp_yolo_cfg.flush()
            with tempfile.NamedTemporaryFile() as tmp_yolo_data:
                for line in fix_yolo_data_path(yolo_data_content):
                    tmp_yolo_data.write(bytes(line+'\n', encoding="utf-8"))
                tmp_yolo_data.flush()
                self.yolo = Detector(bytes(tmp_yolo_cfg.name, encoding="utf-8"), bytes(yolo_weight, encoding="utf-8"), 0, bytes(tmp_yolo_data.name, encoding="utf-8"))
    def detect(self, images: list, depths: list, threshold: float = 0.0, yolo_threshold: float = 0.1, yolo_nms: float = 0.3, max_objs: int = 7):
        with torch.no_grad():
            return predict(self.model, self.yolo, images, depths, threshold=threshold, yolo_thresh=yolo_threshold, nms=yolo_nms, max_objs=max_objs)

def predict_yolo(net, img, n_rect=7, n_rect_search=200, yolo_thresh=0.1, nms=.25):
    img2 = Image(img)
    #detect(self, Image image, float thresh=.5, float hier_thresh=.5, float nms=.45)
    results = [] if net is None else net.detect(img2, thresh=yolo_thresh, nms=nms)
    #results = []
    centers = []
    scores = []
    cats = []
    for cat, score, bounds in results:
        x, y, w, h = bounds
        centers += [[x,y]]
        scores += [score]
        cats += [cat]
    cats = np.asarray(cats)
    centers = np.asarray(centers, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)[:n_rect]
    centers = centers[order]
    scores = scores[order]
    cats = cats[order]
    cats = cats.astype(str)
    if len(centers)==0: # if no object detected, center crop
        return predict_selective_search(img, n_rect=n_rect, n_rect_search=n_rect_search)
    return centers, scores, cats

def predict_selective_search(img, n_rect=7, n_rect_search=200):
    rects = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)[1][:n_rect_search]
    centers = []
    candidates = set()
    for r in rects:
        if r['rect'] in candidates:
            continue
        if r['size'] < 1200:
            continue
        x, y, w, h = r['rect']
        candidates.add(r['rect'])
    candidates = list(candidates)[:n_rect]
    for box in candidates:
        x, y, w, h = box
        centers += [[x+w//2,y+h//2]]
    if len(centers)==0: # if no object detected, center crop
        centers = np.array([[img.shape[1]//2, img.shape[0]//2]], dtype=np.int32)
    return np.asarray(centers, dtype=np.int32), np.zeros(len(centers), dtype=np.float32), np.asarray(['unk']*len(centers))

def preprocess_raw(img_, depth_, centers):
    imgs = []
    for center in centers:
        img = np.copy(img_)
        depth = np.copy(depth_)
        d = normalize_depth(crop_image(depth, center, crop_size=cfg.crop_size))
        img = crop_image(img, center, crop_size=cfg.crop_size)
        img[...,-1] = d
        img = preprocess_input(img)
        imgs += [img]
    return imgs

def bbox_postprocess(bbox, center):
    crop_b = cfg.crop_size//2
    left_c = center[0]-crop_b
    up_c = center[1]-crop_b
    bbox = bbox*cfg.crop_size
    bbox[...,0] += left_c
    bbox[...,1] += up_c
    return bbox

def inference_bbox(model, inputs, threshold=0.0):
    inp = torch.from_numpy(np.transpose(inputs, (0, 3, 1, 2))) # (b, h, w, c) -> (b, c, h, w)
    m = next(model.parameters())
    if m.is_cuda:
        inp = inp.cuda()
    inp = inp.type_as(m)
    model.eval()
    with torch.no_grad():
        bboxes, degs, confs = feature2bboxwdeg(model(inp).detach().cpu().numpy(), threshold)
    return bboxes, degs, confs

def predict(model, yolo_det, rgbs, depths, threshold=0.0, yolo_thresh=0.1, max_objs=7, nms=.25):
    model.eval()
    with torch.no_grad():
        rgds = []
        centers = []
        cats = []
        scores = []
        splits = []
        for rgb, depth in zip(rgbs, depths):
            center, score, cat = predict_yolo(yolo_det, rgb, yolo_thresh=yolo_thresh, n_rect=max_objs, nms=nms)
            rgd = preprocess_raw(rgb, depth, center)
            rgds += list(rgd)
            centers += list(center)
            cats += list(cat)
            scores += list(score)
            splits += [len(centers)]
        if len(splits)>0:
            del splits[-1]
        rgds = np.asarray(rgds,dtype=np.float32)
        bboxes, degs, confs = inference_bbox(model, rgds, threshold=threshold)
        new_bboxes, new_degs, new_confs, new_centers, new_cats, new_scores = [], [], [], [], [], []
        for bbox, deg, conf, center, cat, score in zip(np.split(bboxes, splits), np.split(degs, splits), np.split(confs, splits), np.split(centers, splits), np.split(cats, splits), np.split(scores, splits)):
            empty_id = []
            for i in range(len(center)):
                bbox[i] = [ bbox_postprocess(b, center[i]) for b in bbox[i]  ]
                if len(bbox[i])==0:
                    empty_id += [i]
            bbox = np.delete(bbox, empty_id, 0)
            deg = np.delete(deg, empty_id, 0)
            conf = np.delete(conf, empty_id, 0)
            center = np.delete(center, empty_id, 0)
            score = np.delete(score, empty_id, 0)
            cat = np.delete(cat, empty_id, 0)
            i = np.argsort([-len(x) for x in bbox])
            new_bboxes += [bbox[i]]
            new_degs += [deg[i]]
            new_confs += [conf[i]]
            new_centers += [center[i]]
            new_scores += [score[i]]
            new_cats += [cat[i]]
    return new_bboxes, new_degs, new_confs, new_centers, new_cats, new_scores


