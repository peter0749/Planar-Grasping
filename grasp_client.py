import io
import numpy as np
import json
import subprocess
import base64

def grasp2vec(x):
    x_mean = x.mean(axis=0)
    x_up = (x[0]+x[1])/2.0
    v = x_up-x_mean
    v_len = np.linalg.norm(v)
    return (x_mean, v/v_len)

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
        return json.loads(response)
    def best_grasp(self, img, depth):
        return np.asarray(self.get(img, depth)['grasp'][0][0])
    def get_topK(self, img, depth, K=5):
        all_grasp = []
        all_score = []
        results = self.get(img, depth)
        grasps = results['grasp']
        scores = results['confidence']
        for i, grasp_c in enumerate(grasps):
            for j, grasp_b in enumerate(grasp_c):
                all_grasp.append( grasp_b )
                all_score.append( -scores[i][j] )
        all_grasp = np.asarray(all_grasp)
        return all_grasp[np.argsort(all_score)[:K]]
    def __del__(self):
        self.sub_process.kill()

if __name__=='__main__':
    import cv2
    import matplotlib.pyplot as plt
    test = GraspHandler()
    paths = [
                ('example_imgs/rgb_image.png', 'example_imgs/depth_image.npy'),
                ('example_imgs/rgb_image2.png', 'example_imgs/depth_image2.npy'),
                ('example_imgs/rgb1.jpg', 'example_imgs/depth1.npy'),
                ]
    for (img_p,depth_p) in paths:
        img = cv2.imread(img_p, cv2.IMREAD_COLOR)[...,::-1]
        depth = np.load(depth_p, allow_pickle=True, fix_imports=True)
        pts = test.get_topK(img,depth,K=5)
        img_ = np.copy(img)
        cv2.polylines(img_, pts.astype(np.int32), True, (255,0,0), 5)
        vecs = [ grasp2vec(pt) for pt in pts ]
        for vec in vecs:
            cv2.line(img_, tuple(vec[0].astype(np.int32)), tuple((vec[0]+vec[1]*(img_.shape[0]/20.)).astype(np.int32)), (0,0,255), 5)
        plt.imshow(img_)
        plt.show()
    del test

