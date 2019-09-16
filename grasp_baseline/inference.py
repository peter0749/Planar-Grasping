import os
import re
import tempfile
import selectivesearch
import torch
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
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

class SingleObjectGraspDetector(object):
    def __init__(self, backbone: str = 'vgg19_bn', weight_path: str = os.path.join(os.path.split(__file__)[0],"weights","grasp_model.pth"), f16: bool = False, cuda: bool = True
            ,**kwargs):
        self.model = GraspModel(backbone=backbone, with_fc=False)
        if cuda:
            self.model = self.model.cuda()
        state_dict = torch.load(weight_path) if cuda else torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.half() if f16 else self.model.float()
        self.model.eval()
    def detect(self, image, depth, bbox, threshold: float = 0.0):
        with torch.no_grad():
            return predict_single_object(self.model, image, depth, bbox, threshold=threshold)

class GraspDetector(object):
    def __init__(self, backbone: str = 'vgg19_bn', weight_path: str = os.path.join(os.path.split(__file__)[0],"weights","grasp_model.pth"), f16: bool = False, cuda: bool = True,
            yolo_cfg: str = os.path.join(os.path.split(__file__)[0],"cfg","yolov3-spp.cfg"), yolo_weight: str = os.path.join(os.path.split(__file__)[0],"weights","yolov3-spp.weights"), yolo_data: str = os.path.join(os.path.split(__file__)[0],"cfg","coco.data")
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
        try:
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
        except:
            self.yolo = None
    def detect(self, images: list, depths: list, threshold: float = 0.0, yolo_threshold: float = 0.1, yolo_nms: float = 0.25, max_objs: int = 7):
        with torch.no_grad():
            return predict(self.model, self.yolo, images, depths, threshold=threshold, yolo_thresh=yolo_threshold, nms=yolo_nms, max_objs=max_objs)

def predict_yolo(net, img, n_rect=7, n_rect_search=200, yolo_thresh=0.1, nms=.25):
    img2 = Image(img)
    #detect(self, Image image, float thresh=.5, float hier_thresh=.5, float nms=.45)
    results = [] if net is None else net.detect_foreground(img2, thresh=yolo_thresh, nms=nms)
    #results = []
    centers = []
    scores = []
    for score, bounds in results:
        x, y, w, h = bounds
        centers += [[x-w/2,y-h/2,x+w/2,y+h/2]]
        scores += [score]
    centers = np.asarray(centers, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores)[:n_rect]
    centers = centers[order]
    scores = scores[order]
    if len(centers)==0: # if no object detected
        return predict_selective_search(img, n_rect=n_rect, n_rect_search=n_rect_search)
    return centers, scores

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
        centers += [[x,y,x+w,y+h]]
    if len(centers)==0: # if no object detected, center crop
        tx1 = (img.shape[1]-cfg.crop_size)/2
        ty1 = (img.shape[0]-cfg.crop_size)/2
        tx2 = (img.shape[1]+cfg.crop_size)/2
        ty2 = (img.shape[0]+cfg.crop_size)/2
        centers = np.array([[tx1,ty1,tx2,ty2]], dtype=np.float32)
    return np.asarray(centers, dtype=np.float32), np.zeros(len(centers), dtype=np.float32)

def preprocess_raw(img_, depth_, centers):
    imgs = []
    for center in centers:
        cx, cy = int((center[0]+center[2])//2), int((center[1]+center[3])//2)
        img = np.copy(img_)
        depth = np.copy(depth_)
        d = normalize_depth(crop_image(depth, (cx,cy), crop_size=cfg.crop_size))
        img = crop_image(img, (cx,cy), crop_size=cfg.crop_size)
        img[...,-1] = d
        img = preprocess_input(img)
        imgs += [img]
    return imgs

def bbox_postprocess(bbox, center):
    cx, cy = int((center[0]+center[2])//2), int((center[1]+center[3])//2)
    crop_b = cfg.crop_size//2
    left_c = cx-crop_b
    up_c = cy-crop_b
    bbox = bbox*cfg.crop_size
    bbox[...,0] += left_c
    bbox[...,1] += up_c
    ## TODO: Add try ... except clause
    p1 = Polygon(bbox)
    p2 = Polygon([ [center[0],center[1]], [center[2],center[1]], [center[2],center[3]], [center[0],center[3]]  ])
    if p1.intersects(p2):
        insc = p1.intersection(p2).area
        if insc/p1.area>0.75:
            return bbox
    return None

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

def predict_single_object(model, rgb, depth, bbox, threshold=0.0):
    model.eval()
    rgd = preprocess_raw(rgb, depth, [bbox])
    with torch.no_grad():
        grasps, _, confs = inference_bbox(model, rgd, threshold=threshold)
    grasps = grasps[0]
    confs = confs[0]
    new_grasps = []
    new_confs = []
    for g,c in zip(grasps,confs):
        grasp_postprocessed = bbox_postprocess(g, bbox)
        if grasp_postprocessed is None:
            continue
        new_grasps.append(grasp_postprocessed)
        new_confs.append(c)
    return np.asarray(new_grasps), np.asarray(new_confs)

def predict(model, yolo_det, rgbs, depths, threshold=0.0, yolo_thresh=0.1, max_objs=7, nms=.25):
    model.eval()
    with torch.no_grad():
        rgds = []
        centers = []
        scores = []
        splits = []
        for rgb, depth in zip(rgbs, depths):
            center, score = predict_yolo(yolo_det, rgb, yolo_thresh=yolo_thresh, n_rect=max_objs, nms=nms)
            rgd = preprocess_raw(rgb, depth, center)
            rgds += list(rgd)
            centers += list(center)
            scores += list(score)
            splits += [len(centers)]
        if len(splits)>0:
            del splits[-1]
        rgds = np.asarray(rgds,dtype=np.float32)
        bboxes, degs, confs = inference_bbox(model, rgds, threshold=threshold)
        new_bboxes, new_degs, new_confs, new_centers, new_scores = [], [], [], [], []
        for bbox, deg, conf, center, score in zip(np.split(bboxes, splits), np.split(degs, splits), np.split(confs, splits), np.split(centers, splits), np.split(scores, splits)):
            bbox = list(bbox)
            deg = list(deg)
            conf = list(conf)
            empty_id = []
            for i in range(len(center)):
                bbox_ = []
                deg_ = []
                conf_ = []
                for b,d,c in zip(bbox[i],deg[i],conf[i]):
                    processed_sub_bbox = bbox_postprocess(b, center[i])
                    if not processed_sub_bbox is None:
                        bbox_.append(processed_sub_bbox)
                        deg_.append(d)
                        conf_.append(c)
                bbox_ = np.asarray(bbox_, dtype=np.float32)
                deg_  = np.asarray(deg_,  dtype=np.float32)
                conf_ = np.asarray(conf_, dtype=np.float32)
                conf_od = np.argsort(-conf_) # sort by grasping confidence
                #conf_od = np.arange(len(conf_)).astype(np.int32)
                bbox[i] = bbox_[conf_od]
                deg[i] = deg_[conf_od]
                conf[i] = conf_[conf_od]
                if len(bbox[i])==0:
                    empty_id += [i]
            bbox = np.delete(bbox, empty_id, 0)
            deg = np.delete(deg, empty_id, 0)
            conf = np.delete(conf, empty_id, 0)
            center = np.delete(center, empty_id, 0)
            score = np.delete(score, empty_id, 0)
            i = np.argsort([-np.max(x) for x in conf])
            new_bboxes += [bbox[i]]
            new_degs += [deg[i]]
            new_confs += [conf[i]]
            new_centers += [center[i]]
            new_scores += [score[i]]
    return new_bboxes, new_degs, new_confs, new_centers, new_scores
