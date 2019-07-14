import config as cfg
import numpy as np
import torch

def grasp_loss(inputs, target):
    # inputs: (b,7,7,7) -- confidence, x, y, w, h, cos(2a), sin(2a)
    # target: (b,7,7,7) -- same as above

    inputs = inputs.permute(0, 2, 3, 1) # (b,c,h,w) -> (b,h,w,c)
    target = target.permute(0, 2, 3, 1) # (b,c,h,w) -> (b,h,w,c)

    lambda_coord = cfg.lambda_coord
    lambda_rot = cfg.lambda_rot
    m = cfg.hinge_margin

    input_heatmap = inputs[...,0:1]
    input_xy = inputs[...,1:3]
    input_wh = inputs[...,3:5]
    input_cos_sin = inputs[...,5:7]

    gt_heatmap = target[...,0:1] # (-1,1)
    gt_mask = (gt_heatmap>0).type_as(inputs) # (0, 1)
    gt_xy   = target[...,1:3]
    gt_wh   = target[...,3:5]
    gt_cos_sin = target[...,5:7]

    xy_loss = torch.abs((gt_xy-input_xy)*gt_mask).sum(-1).sum(-1).sum(-1) # shape: (b,)
    wh_loss = torch.abs((gt_wh-input_wh)*gt_mask).sum(-1).sum(-1).sum(-1) # shape: (b,)
    coord_loss = xy_loss+wh_loss # shape: (b,)

    rot_loss = torch.abs((gt_cos_sin-input_cos_sin)*gt_mask).sum(-1).sum(-1).sum(-1) # shape: (b,)

    conf_loss = (m-input_heatmap*gt_heatmap).clamp(min=0) # hinge loss: max{0, 1-y_gt*y_hat}**2
    conf_loss = conf_loss.sum(-1).sum(-1).sum(-1) # shape: (b,)

    loss = lambda_coord*coord_loss + lambda_rot*rot_loss + conf_loss # shape: (b,)
    loss[loss!=loss] = 0
    loss[loss==np.inf] = 0
    return loss.mean()

if __name__=='__main__':
    i = torch.randn((3,7,7,7))
    t = torch.randn((3,7,7,7))
    loss = grasp_loss(i,t)
    print(loss)
