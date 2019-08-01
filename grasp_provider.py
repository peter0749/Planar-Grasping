#!/usr/bin/env /home/peter/anaconda3/envs/dnn/bin/python
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#sys.stdin = open("/dev/stdin")
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from grasp_baseline.inference import GraspDetector
detector = GraspDetector(cuda=True)
# block input:
for line in sys.stdin:
    paths = line.strip().split()
    if len(paths)!=2:
        continue
    img_path, depth_path = paths
    img = cv2.imread(img_path.strip(), cv2.IMREAD_COLOR)[...,::-1]
    depth = np.load(depth_path.strip(), allow_pickle=True, fix_imports=True)
    bboxes, degs, confs, centers, cats, scores = detector.detect([img], [depth], threshold=-2, yolo_threshold=0.05)

    result_str = ''
    try:
        best_grasp = bboxes[0][0][0]
        grasp_center = np.mean(best_grasp, axis=0)
        result_str = json.dumps(grasp_center.tolist(), ensure_ascii=True)
    except:
        pass
    print(result_str)
    sys.stdout.flush()
