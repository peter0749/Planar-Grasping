import os
import sys
import numpy as np
import glob
from grasp_baseline import utils
from grasp_baseline import config as cfg
from grasp_baseline.dataset import *
from tqdm import tqdm

if __name__=='__main__':
    ids = get_cornell_grasp_ids()
    for id_ in tqdm(ids):
        img_p, pcl_p, pos_p, neg_p, pcl_npy, pos_npy, neg_npy = cornell_grasp_id2realpath(id_)
        pts, indices = parse_pcl(pcl_p)
        np.savez(pcl_npy, pts=pts, indices=indices)
        pos = parse_bbox(pos_p)
        np.save(pos_npy, pos)
        neg = parse_bbox(neg_p)
        np.save(neg_npy, neg)

