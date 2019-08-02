#!/usr/bin/env /home/test/grasp_models_pyvenv/bin/python
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#sys.stdin = open("/dev/stdin")
import io
import base64
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
    img_b64, depth_b64 = paths
    img_b64 = img_b64.strip()
    depth_b64 = depth_b64.strip()
    img = np.load(io.BytesIO(base64.b64decode(img_b64)), allow_pickle=True, fix_imports=True)
    depth = np.load(io.BytesIO(base64.b64decode(depth_b64)), allow_pickle=True, fix_imports=True)
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
