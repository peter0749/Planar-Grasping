### Pre-requisites

Python>=3.5 (3.6.1 would be best)

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

Download and decompress pre-train weights to `${This REPO}/grasp_baseline/weights`

[weights](https://drive.google.com/drive/folders/11KUFxY68539TQKutcn_1fgeVH2mkS2K0)

### Usage

```
### In any python script
from grasp_baseline.inference import GraspDetector
detector = GraspDetector()
grasps, degrees, graspscores, object_bounding_boxes, categories, yolo_scores = detector.detect([img1, img2, ...], [depth1, depth2, ...])
```

Where the first augment is a list of RGB images, the second is a list of depth images.

### Visualize

```
visualize([img1, img2, ...], grasps, graspscores, object_bounding_boxes, categories)
```

### Example

[inference_test.ipynb](./inference_test.ipynb)

### Test on RUST

```
cd cargo_test
cargo run
```
