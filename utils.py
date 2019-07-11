import config as cfg
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from shapely.geometry import Polygon

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def get_imgaug():
    seq_spatial = sometimes(iaa.Affine(
                translate_px={"x": (-60, 60), "y": (-60, 60)}, # translate by -20 to +20 percent (per axis)
                rotate=(-175, 175), # rotate by -160 to +160 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
    seq_color = sometimes(iaa.Sequential(
        [
            iaa.SomeOf((1, 3),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    # search either for all edges or for directed edges,
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.Add((-5, 5), per_channel=0.5),
                ],
                random_order=True
            )
        ],
        random_order=True
    ))
    return seq_spatial, seq_color

def center_crop(x, pts=None, crop_size=320):
    crop_b = crop_size//2
    center_x = x.shape[1]//2
    center_y = x.shape[0]//2
    left_c = max(0, center_x-crop_b)
    right_c = left_c+crop_size
    up_c = max(0, center_y-crop_b)
    down_c = up_c+crop_size
    if not pts is None:
        pts[...,0] -= left_c
        pts[...,1] -= up_c
        return x[up_c:down_c, left_c:right_c], pts
    return x[up_c:down_c, left_c:right_c]
def preprocess_input(x):
    x = np.concatenate( [ cv2.resize(x[...,i], (cfg.input_size, cfg.input_size), interpolation=cv2.INTER_AREA)[...,np.newaxis] for i in range(x.shape[-1])  ] , axis=-1)
    x = x.astype(np.float32)/255.0-np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32) # [0,255] -> [0,1] -> -mean
    x = x / np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32) # /std
    return x
def aug_data(seq, img, pts=None):
    if pts is None:
        return seq(image=img)
    pts_shape = pts.shape
    kps = [ Keypoint(x=x[0], y=x[1]) for x in pts.reshape(-1,2) ]
    kpsoi = KeypointsOnImage(kps, shape=img.shape)
    img_auged, kpsoi_auged = seq(image=img, keypoints=kpsoi)
    res_pts = kpsoi_auged.to_xy_array().reshape(pts_shape)
    return img_auged, res_pts

def feature2bboxwdeg(p, th):
    # p: (b, c, h, w)
    c, x, y, w, h, cos2a, sin2a = [ t[:,0] for t in np.split(p, 7, axis=1) ] # 7*(b,h,w)
    batch_size = c.shape[0]
    bboxes = []
    degs = []
    for b in range(batch_size):
        pos = c[b]>th # (h,w)
        argpos = np.argwhere(pos)
        xx = np.asarray([ idx[1]+x[b, idx[0], idx[1]] for idx in argpos ])
        yy = np.asarray([ idx[0]+y[b, idx[0], idx[1]] for idx in argpos ])
        ww = np.asarray([ w[b, idx[0], idx[1]] for idx in argpos ])
        hh = np.asarray([ h[b, idx[0], idx[1]] for idx in argpos ])
        c2a = np.asarray([ cos2a[b, idx[0], idx[1]] for idx in argpos ])
        s2a = np.asarray([ sin2a[b, idx[0], idx[1]] for idx in argpos ])
        tan_a = (1-c2a) / s2a
        tan_a[tan_a!=tan_a] = 0
        aa = np.arctan(tan_a)
        N  = len(aa)
        v = np.array([[-hh/2, ww/2], # v0
                      [ hh/2, ww/2], # v1
                      [ hh/2,-ww/2], # v2
                      [-hh/2,-ww/2]]) # v3 total:(4, 2, N)
        v = np.transpose(v, (2,1,0)) # (N, 2, 4)

        R = np.array([
                [np.cos(aa), -np.sin(aa)],
                [np.sin(aa),  np.cos(aa)]
            ]) # (2, 2, N)
        R = np.transpose(R, (2,0,1)) # (N, 2, 2)
        bs = []
        for n in range(N):
            bs += [(R[n]@v[n]).T] # (2,2) x (2,4) = (2,4) -> (4,2)
        bs = np.asarray(bs) # (N, 4, 2)
        if len(bs)>0:
            bs[...,1] = -bs[...,1] # from normal to upside down (for image format)
            bs[...,0] += xx[...,np.newaxis]
            bs[...,1] += yy[...,np.newaxis]
            bs /= cfg.grid_size
        bboxes += [bs]
        degs += [aa]
    return bboxes, degs

def bbox_correct(preds, gt):
    correct = 0
    for p in preds:
        p_poly = Polygon( [ [x[0],x[1]] for x in p ]  )
        v_p = p[1]-p[0]
        for bbox in gt:
            g_poly = Polygon( [ [x[0],x[1]] for x in bbox ]  )
            g_p = bbox[1]-bbox[0]
            deg_diff = np.abs(np.degrees(np.arccos((v_p*g_p).sum() / (np.linalg.norm(v_p)*np.linalg.norm(g_p)+1e-8))))
            deg_diff_m180 = np.abs(deg_diff-180) # robotic arms are symmeric
            try:
                inter = p_poly.intersection(g_poly).area
                union = p_poly.area+g_poly.area-inter
                iou = inter/(union+1e-8)
            except:
                continue
            if ( deg_diff<30 or deg_diff_m180<30 ) and iou>0.25:
                correct += 1
                break
    return correct

if __name__=='__main__':
    get_R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    bbox = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    a = (get_R(np.radians(30+180))@bbox.T).T[np.newaxis]
    b = (get_R(np.radians(30))@bbox.T).T[np.newaxis]
    print(a)
    print(b)
    print(bbox_correct(a,b))
