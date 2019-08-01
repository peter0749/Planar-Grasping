#!/usr/bin/env /home/test/grasp_models_pyvenv/bin/python
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from grasp_baseline.inference import GraspDetector
detector = GraspDetector(cuda=False)
# block input:
for line in sys.stdin:
    paths = line.strip().split()
    if len(paths)!=2:
        break
    img_path, depth_path = paths
    img = cv2.imread(img_path.strip(), cv2.IMREAD_COLOR)[...,::-1]
    depth = np.load(depth_path.strip(), allow_pickle=True, fix_imports=True)
    bboxes, degs, confs, centers, cats, scores = detector.detect([img], [depth], threshold=-2, yolo_threshold=0.05)

    result_str = '-1 -1'
    try:
        best_grasp = bboxes[0][0][0]
        grasp_center = np.mean(best_grasp, axis=0)
        result_str = '{:.2f} {:.2f}'.format(grasp_center[0], grasp_center[1])
    except:
        pass
    print(result_str)
    sys.stdout.flush()
