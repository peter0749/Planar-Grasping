import os
import sys
import time
import datetime
import gc
import config as cfg
import argparse
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
from tensorboardX import SummaryWriter
from loss import grasp_loss
from dataset import CornellGraspDataset
from models import GraspModel
from utils import bbox_correct, feature2bboxwdeg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=32, help='input batch size'
    )
    parser.add_argument(
        '--nepoch', type=int, default=100, help='number of epochs to train for'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0005, help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.001, help='Weight decay (l2)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='Momentum of SGD'
    )
    parser.add_argument(
        '--backbone', type=str, default='vgg16', help='Model backbone (vgg16/resnet50)'
    )
    parser.add_argument(
        '--n_workers', type=int, default=5, help='Multiprocessing'
    )
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--f16', action='store_true', help='Half precision training')
    parser.add_argument('--with_fc', action='store_true', help='With fc layers (like YOLOv1)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    return args

def main(args):
    dataset = CornellGraspDataset(return_box=True)
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_workers,
                pin_memory=True if args.cuda else False,
                drop_last=True
                )

    begin_ts = time.time()
    fold_acc = np.zeros(cfg.n_folds)
    for fold_id in range(cfg.n_folds):
        model = GraspModel(backbone=args.backbone, with_fc=args.with_fc)
        if args.f16:
            model = model.half()
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
        if args.cuda:
            model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45,90], gamma=0.2)

        dataset.set_current_fold(fold_id)
        for e in range(args.nepoch):
            model.train()
            dataset.set_current_state('train')
            cnt_accum = 0
            bbox_accum = 0
            loss_accum = 0
            acc_accum = 0
            for (inp, target, bbox) in dataloader:
                if args.f16:
                    inp = inp.half()
                    target = target.half()
                if args.cuda:
                    inp = inp.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                out = model(inp)
                loss = grasp_loss(out, target)
                loss.backward()
                optimizer.step()
                loss_accum += loss.item() * len(inp)
                bbox_pred = feature2bboxwdeg(out.detach().cpu().numpy(), cfg.threshold)[0]
                bbox = bbox.numpy()
                for aa, bb in zip(bbox_pred, bbox):
                    acc_accum += bbox_correct(aa,bb)
                    bbox_accum += len(aa)
                cnt_accum += len(inp)
            avg_train_loss = loss_accum / cnt_accum
            avg_train_acc = acc_accum / (bbox_accum+1e-8)

            model.eval()
            dataset.set_current_state('test')
            cnt_accum = 0
            bbox_accum = 0
            loss_accum = 0
            acc_accum = 0
            with torch.no_grad():
                for (inp, target, bbox) in dataloader:
                    if args.f16:
                        inp = inp.half()
                        target = target.half()
                    if args.cuda:
                        inp = inp.cuda()
                        target = target.cuda()
                    out = model(inp)
                    loss = grasp_loss(out, target)
                    loss_accum += loss.item() * len(inp)
                    bbox_pred = feature2bboxwdeg(out.detach().cpu().numpy(), cfg.threshold)[0]
                    bbox = bbox.numpy()
                    for aa, bb in zip(bbox_pred, bbox):
                        acc_accum += bbox_correct(aa,bb)
                        bbox_accum += len(aa)
                    cnt_accum += len(inp)
                avg_val_loss = loss_accum / cnt_accum
                avg_val_acc = acc_accum / (bbox_accum+1e-8)

            lr_scheduler.step()
            end_ts = time.time()
            elapsed_sec = end_ts-begin_ts
            elapsed = datetime.timedelta(seconds=int(elapsed_sec))
            nepochs_passed = args.nepoch*fold_id+e+1
            nepochs_remain = args.nepoch*cfg.n_folds - nepochs_passed
            eta = datetime.timedelta(seconds=int(elapsed_sec/nepochs_passed*nepochs_remain))
            fold_acc[fold_id] = avg_val_acc
            print("Fold: [%2d/%2d] Epoch: [%2d/%2d] loss: %.4f val_loss: %.4f acc: %.4f val_acc: %.4f elapsed: %s, eta: %s"%(fold_id+1, cfg.n_folds, e+1, args.nepoch, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, str(elapsed), str(eta)))
            sys.stdout.flush()
        del model
        gc.collect()
        torch.cuda.empty_cache() # clear memory after every fold
    print("ACC: %.4f, STD: %.4f"%( np.mean(avg_val_acc), np.std(avg_val_acc)  ))
    sys.stdout.flush()

if __name__=='__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('forkserver', force=True)
    args = parse_args()
    print(args)
    main(args)

