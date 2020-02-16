from . import config as cfg
import numpy as np
import torch
import torch.nn.functional as F

def IEMD(h, h_hat, gamma=0.3):
    # shape: (B, N, D)
    h_len = h.size(-1) # D

    h_circular = torch.cat((h, h), -1) # shape: (B, N, 2*D)
    h_hat_circular = torch.cat((h_hat, h_hat), -1) # shape: (B, N, 2*D)
    c = torch.cumsum(h_circular, -1) # shape: (B, N, 2*D)
    c_hat = torch.cumsum(h_hat_circular, -1) # shape: (B, N, 2*D)

    ws = torch.empty_like(h) # (B, N, D)
    for i in range(1, h_len+1):
        c_i_to_ipt = c[...,i:i+h_len] - c[...,i-1:i] # shape: (B, N, D)
        c_hat_i_to_ipt = c_hat[...,i:i+h_len] - c_hat[...,i-1:i] # shape: (B, N, D)
        w_i_t = (c_i_to_ipt - c_hat_i_to_ipt)
        w_i_t_2 = w_i_t*w_i_t
        w_i = w_i_t_2.mean(-1)# shape: (B, N)
        ws[...,i-1] = w_i
    return ws**gamma # better slope

def grasp_loss(inputs, target, gamma=0.3):
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
    input_angle = inputs[...,5:]

    gt_heatmap = target[...,0:1] # (-1,1)
    gt_mask = (gt_heatmap>0).type_as(inputs) # (0, 1)
    gt_xy   = target[...,1:3]
    gt_wh   = target[...,3:5]
    gt_angle = target[...,5:]

    xy_loss = torch.abs((gt_xy-input_xy)*gt_mask).sum(-1).sum(-1).sum(-1) # shape: (b,)
    wh_loss = torch.abs((gt_wh-input_wh)*gt_mask).sum(-1).sum(-1).sum(-1) # shape: (b,)
    coord_loss = xy_loss+wh_loss # shape: (b,)

    #rot_loss = torch.abs((gt_cos_sin-input_cos_sin)*gt_mask).sum(-1).sum(-1).sum(-1) # shape: (b,)
    #rot_loss = (F.binary_cross_entropy(input_angle, gt_angle, reduction='none')*gt_mask).sum(-1).sum(-1).sum(-1)
    #rot_loss = (IEMD(gt_angle, input_angle, gamma=gamma)*gt_mask).sum(-1).sum(-1).sum(-1)
    rot_loss = ((1.0-gt_angle*input_angle)*gt_mask).sum(-1).sum(-1).sum(-1)

    conf_loss = (m-input_heatmap*gt_heatmap).clamp(min=0) # hinge loss: max{0, 1-y_gt*y_hat}**2
    conf_loss = conf_loss.sum(-1).sum(-1).sum(-1) # shape: (b,)

    loss = lambda_coord*coord_loss + lambda_rot*rot_loss + conf_loss # shape: (b,)
    loss = torch.clamp(loss, 0, 1000)
    return loss.mean()

if __name__=='__main__':
    i = torch.randn((3,7,7,7))
    t = torch.randn((3,7,7,7))
    loss = grasp_loss(i,t)
    print(loss)
