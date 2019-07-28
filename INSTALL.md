### Pre-requisites

Python>=3.5

Python3-dev

OpenCV==3.4.0

shapely

PyTorch

numba

selectivesearch

matplotlib

[Darknet](https://github.com/pjreddie/darknet)

[yolo34py](https://github.com/madhawav/YOLO3-4-Py) (pydarknet)

### INSTALL

Install dependencies:

```
### First install darknet and yolo34py
pip install shapely, numba, opencv-python==3.4.0, numba, selectivesearch
cd ${This REPO}
pip install -e .
```

This will install `grasp_baseline`

### Usage

```
### In any python script
from grasp_baseline.inference import GraspDetector
detector = GraspDetector()
grasps, degrees, object_bounding_boxes, categories, graspscores = detector.detect([img1, img2, ...], [depth1, depth2, ...])
```

Where the first augment is a list of RGB images, the second is a list of depth images.

### Visualize

```
visualize([img1, img2, ...], grasps, graspscores, object_bounding_boxes, categories)
```

### Example

[inference_test.ipynb](./inference_test.ipynb)