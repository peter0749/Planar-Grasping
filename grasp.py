#!/usr/bin/env /home/test/grasp_models_pyvenv/bin/python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from grasp_baseline.inference import GraspDetector
detector = GraspDetector(cuda=False)
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)[...,::-1]
depth = np.load(sys.argv[2], allow_pickle=True, fix_imports=True)
#depth = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
bboxes, degs, confs, centers, cats, scores = detector.detect([img], [depth], threshold=-2, yolo_threshold=0.05)

#from grasp_baseline.utils import visualize
#visualize([img], bboxes, confs, centers, cats)

result_str = '-1 -1'
try:
    best_grasp = bboxes[0][0][0]
    grasp_center = np.mean(best_grasp, axis=0)
    result_str = '{:.2f} {:.2f}'.format(grasp_center[0], grasp_center[1])
except:
    pass
print(result_str)
