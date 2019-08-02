import io
import numpy as np
import json
import subprocess
import base64

class GraspHandler(object):
    def __init__(self, grasp_provider="./grasp_provider.py"):
        self.sub_process = subprocess.Popen((grasp_provider,), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    def get(self, img, depth):
        with io.BytesIO() as fp:
            np.save(fp, img, allow_pickle=True, fix_imports=True)
            img_b64 = base64.b64encode(fp.getvalue()).decode("utf-8")
        with io.BytesIO() as fp:
            np.save(fp, depth, allow_pickle=True, fix_imports=True)
            depth_b64 = base64.b64encode(fp.getvalue()).decode("utf-8")

        command = img_b64 + " " + depth_b64 + "\n"
        self.sub_process.stdin.write(command.encode())
        response = self.sub_process.stdout.readline().decode("utf-8")
        try:
            response_dict = json.loads(response)
            best_grasp = np.asarray(response_dict['grasp'][0][0][0], dtype=np.float32)
            best_grasp_center = best_grasp.mean(axis=0)
            result = best_grasp_center
        except:
            result = np.array([],dtype=np.float32)
        return result
    def __del__(self):
        self.sub_process.kill()

if __name__ == '__main__':
    import cv2
    test = GraspHandler()
    paths = [
            ('example_imgs/rgb_image.png', 'example_imgs/depth_image.npy'),
            ('example_imgs/rgb_image2.png', 'example_imgs/depth_image2.npy'),
            ]
    for _ in range(10):
        for (img_p,depth_p) in paths:
            img = cv2.imread(img_p, cv2.IMREAD_COLOR)[...,::-1]
            depth = np.load(depth_p, allow_pickle=True, fix_imports=True)
            print(test.get(img,depth))
    del test

