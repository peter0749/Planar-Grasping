from . import config as cfg
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from shapely.geometry import Polygon, MultiPoint
import matplotlib.pyplot as plt

sometimes = lambda aug: iaa.Sometimes(0.6, aug)

def get_imgaug():
    seq_spatial = iaa.Affine(
                translate_px={"x": (-70, 70), "y": (-70, 70)}, # translate by -20 to +20 percent (per axis)
                rotate=(-175, 175), # rotate by -170 to +170 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(30, 200),
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
    seq_color = sometimes(iaa.Sequential(
        [
            iaa.SomeOf((1, 3),
                [
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        iaa.MedianPooling((1, 3)),
                    ]),
                    # search either for all edges or for directed edges,
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.ChannelShuffle(0.1),
                    iaa.JpegCompression(compression=(50, 90)),
                    iaa.Grayscale(alpha=(0.0, 0.4)),
                ],
                random_order=True
            )
        ],
        random_order=True
    ))
    return seq_spatial, seq_color

def visualize(imgs, bboxes, confs, objs):
    for i, img_c in enumerate(imgs):
        for xx,cc,b in zip(bboxes[i], confs[i], objs[i]):
            img_with_bbox = np.copy(img_c[...,::-1])
            cmax = np.max(cc)
            cmin = np.min(cc)
            for x,c in zip(xx,cc):
                cval = int((c-cmin+1)/(cmax-cmin+1)*7)
                cv2.polylines(img_with_bbox, [np.round(np.asarray(x)).astype(np.int32)], True, (255, 255, 0), cval)
            b = b.astype(np.int32)
            ul = [b[0],b[1]]
            ur = [b[2],b[1]]
            dl = [b[0],b[3]]
            dr = [b[2],b[3]]
            bb = np.asarray([ ul, ur, dr, dl ], dtype=np.int32)
            cv2.polylines(img_with_bbox, [bb], True, (0, 0, 255), 3)
            img_with_bbox = img_with_bbox[...,::-1]
            plt.imshow(img_with_bbox)
            plt.show()

def crop_image(img_, center_xy, pts=None, crop_size=320):
    img = np.copy(img_)
    h, w = img.shape[:2]
    x, y = center_xy
    s_offset = crop_size//2
    l_offset = crop_size-s_offset
    left, right, up, down = x-s_offset, x+l_offset, y-s_offset, y+l_offset
    if not pts is None:
        pts[...,0] -= left
        pts[...,1] -= up
    p_left, p_right, p_up, p_down = max(0, -left), max(0, right-w), max(0, -up), max(0, down-h)
    if p_left+p_right+p_up+p_down>0:
        if len(img.shape)==2:
            img = np.pad(img, ((p_up, p_down),(p_left, p_right)), mode='symmetric')
        else:
            img = np.pad(img, ((p_up, p_down),(p_left, p_right),(0,0)), mode='symmetric')
    left += p_left
    up += p_up
    img = img[up:up+crop_size, left:left+crop_size]
    return img if pts is None else (img, pts)

def center_crop(x, pts=None, crop_size=320):
    '''
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
    '''
    center_x = x.shape[1]//2
    center_y = x.shape[0]//2
    return crop_image(x, (center_x, center_y), pts, crop_size)
def preprocess_input(x):
    if x.shape[:2]!=(cfg.input_size, cfg.input_size):
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
    c, x, y, w, h = [ t[:,0] for t in np.split(p[:,:5], 5, axis=1) ] # 5*(b,h,w)
    angles_prob = p[:,5:] # angles_prob: (b, n_ori, h, w)
    angles = np.argmax(angles_prob, axis=1).astype(np.float32) * cfg.orientation_base # angles: (b, h, w)
    batch_size = c.shape[0]
    bboxes = []
    degs = []
    confs = []
    for b in range(batch_size):
        pos = c[b]>th # (h,w)
        argpos = np.argwhere(pos)
        cc = np.asarray([ c[b, idx[0], idx[1]] for idx in argpos ])
        xx = np.asarray([ idx[1]+x[b, idx[0], idx[1]] for idx in argpos ])
        yy = np.asarray([ idx[0]+y[b, idx[0], idx[1]] for idx in argpos ])
        ww = np.asarray([ w[b, idx[0], idx[1]] for idx in argpos ])
        hh = np.asarray([ h[b, idx[0], idx[1]] for idx in argpos ])
        aa = np.asarray([ angles[b, idx[0], idx[1]] for idx in argpos ])
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
        confs += [cc]
    return bboxes, degs, confs

def bbox_correct(preds, gt, iou_t=0.25, deg_t=30):
    correct = 0
    for p in preds:
        try:
            p_poly = Polygon(p).convex_hull
            if not p_poly.is_valid:
                continue
        except ValueError:
            continue
        v_p = p[1]-p[0]
        for bbox in gt:
            if not bbox.any(): # all zeros (padding elements)
                continue
            try:
                g_poly = Polygon(bbox).convex_hull
                if not g_poly.is_valid:
                    continue
            except ValueError:
                continue
            g_p = bbox[1]-bbox[0]
            deg_diff = np.abs(np.degrees(np.arccos((v_p*g_p).sum() / (np.linalg.norm(v_p)*np.linalg.norm(g_p)+1e-8))))
            deg_diff_m180 = np.abs(deg_diff-180) # robotic arms are symmeric
            iou = 0
            if p_poly.intersects(g_poly):
                try:
                    inter = p_poly.intersection(g_poly).area
                    union = p_poly.area + g_poly.area - inter
                    iou = inter/(union+1e-8)
                except:
                    iou = 0
            if min(deg_diff, deg_diff_m180)<deg_t and iou>iou_t:
                correct += 1
                break
    return correct



if __name__=='__main__':
    from tqdm import tqdm
    get_R = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    bbox = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    print("Rotation test...")
    for d in tqdm(np.arange(0, 360, 0.5)):
        p = (get_R(np.radians(d))@bbox.T).T[np.newaxis]
        bbox_correct(p, bbox[np.newaxis])
    print("Done.")
    print("Random bbox + rotation test...")
    for _ in tqdm(range(10000)):
        bbox = np.random.randn(4,2)
        for d in [30, 60]:
            p = (get_R(np.radians(d))@bbox.T).T[np.newaxis]
            bbox_correct(p, bbox[np.newaxis])
    print("Done.")
    print("Random bbox test...")
    for _ in tqdm(range(10000)):
        a = np.random.randn(4,4,2)
        b = np.random.randn(4,4,2)
        bbox_correct(a, b)
    print("Done.")
    print("Null bbox test...")
    for _ in tqdm(range(10000)):
        a = np.zeros((4,4,2))
        b = np.random.randn(4,4,2)
        bbox_correct(a, b)
        bbox_correct(b, a)
    print("Done.")

