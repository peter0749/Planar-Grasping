import numpy as np
import json
import pexpect

class GraspHandler(object):
    def __init__(self, grasp_provider="./grasp_provider.sh"):
        self.sub_process = pexpect.spawn(grasp_provider, echo=False)
    def get(self, img_path, depth_path):
        command = img_path + " " + depth_path
        self.sub_process.sendline(command)
        response = self.sub_process.readline()
        result = json.loads(response)
        return np.asarray(result)
    def __del__(self):
        self.sub_process.close(force=True)

if __name__ == '__main__':
    test = GraspHandler()
    paths = [
            ('example_imgs/rgb_image.png', 'example_imgs/depth_image.npy'),
            ('example_imgs/rgb_image2.png', 'example_imgs/depth_image2.npy'),
            ]
    for _ in range(10):
        for (img,depth) in paths:
            print(test.get(img,depth))
    del test

