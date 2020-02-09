import os
import sys
import time
import datetime
import gc
from grasp_baseline import config as cfg
import argparse
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from grasp_baseline.loss import grasp_loss
from grasp_baseline.dataset import CornellGraspDataset
from grasp_baseline.models import grasp_model
from grasp_baseline.utils import bbox_correct, feature2bboxwdeg

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
        '--weight_decay', type=float, default=0.1, help='Weight decay'
    )
    parser.add_argument(
        '--batchnorm_momentum', type=float, default=0.2, help='Momentum of Batch Norm running means'
    )
    parser.add_argument(
        '--save_path', type=str, default='./weights', help=''
    )
    parser.add_argument(
        '--split_mode', type=str, default='obj', help='obj/img'
    )
    parser.add_argument(
        '--backbone', type=str, default='vgg19_bn', help='Model backbone (vgg11/vgg16/vgg19/vgg11_bn/vgg16_bn/vgg19_bn/mobilenetv2/resnet18/resnet34/resnet50/resnet101/darknet53)'
    )
    parser.add_argument(
        '--n_workers', type=int, default=1, help='Multiprocessing'
    )
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--f16', action='store_true', help='Half precision training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    return args

def setup_dataloader(dataset, args, mode='train'):
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.n_workers,
                pin_memory=True if args.cuda else False,
                drop_last=(mode=='train')
                )
    return dataloader

def main(args):
    if len(args.save_path)>0 and (not os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    print('===***  Preprocessing ***===')
    sys.stdout.flush()
    dataset = CornellGraspDataset(split_mode=args.split_mode, return_box=True)
    print('Done!')
    sys.stdout.flush()

    begin_ts = time.time()
    fold_top1 = np.zeros(cfg.n_folds)
    fold_top5 = np.zeros(cfg.n_folds)
    fold_top1_easy = np.zeros(cfg.n_folds)
    fold_top5_easy = np.zeros(cfg.n_folds)
    logger = SummaryWriter()
    print('===*** Begin Training ***===')
    sys.stdout.flush()
    for fold_id in range(cfg.n_folds):
        base_model = grasp_model(args.backbone)
        if args.f16:
            base_model = base_model.half()
            for layer in base_model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
        for layer in base_model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.momentum = args.batchnorm_momentum
        if args.cuda:
            base_model = base_model.cuda()
        model = torch.nn.DataParallel(base_model) if args.multi_gpu else base_model
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        ### Setup current fold ###
        dataset.set_current_fold(fold_id)

        best_top1_acc = -np.inf
        best_model_top5 = -np.inf
        best_model_top1_easy = -np.inf
        best_model_top5_easy = -np.inf
        for e in range(args.nepoch):
            dataset.set_current_state('train')
            dataloader = setup_dataloader(dataset, args, mode='train')
            model.train()
            cnt_accum = 0
            loss_accum = 0
            top5_acc_accum = 0
            top1_acc_accum = 0
            top5_acc_accum_easy = 0
            top1_acc_accum_easy = 0
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
                bbox_pred, _, bbox_conf = feature2bboxwdeg(out.detach().cpu().numpy(), -np.inf)
                bbox = bbox.numpy()
                for aa, bb, cc in zip(bbox_pred, bbox, bbox_conf):
                    cc_order = np.argsort(-cc)
                    top5_acc_accum += (bbox_correct(aa[cc_order[:5]],bb,iou_t=cfg.iou_threshold,deg_t=cfg.deg_threshold)>0)
                    top1_acc_accum += bbox_correct(aa[cc_order[:1]],bb,iou_t=cfg.iou_threshold,deg_t=cfg.deg_threshold)
                    top5_acc_accum_easy += (bbox_correct(aa[cc_order[:5]],bb,iou_t=cfg.iou_threshold_easy,deg_t=cfg.deg_threshold_easy)>0)
                    top1_acc_accum_easy += bbox_correct(aa[cc_order[:1]],bb,iou_t=cfg.iou_threshold_easy,deg_t=cfg.deg_threshold_easy)
                cnt_accum += len(inp)
            avg_train_loss = loss_accum / cnt_accum
            avg_train_top5_acc = top5_acc_accum / cnt_accum
            avg_train_top1_acc = top1_acc_accum / cnt_accum
            avg_train_top5_acc_easy = top5_acc_accum_easy / cnt_accum
            avg_train_top1_acc_easy = top1_acc_accum_easy / cnt_accum
            del dataloader

            dataset.set_current_state('test')
            val_dataloader = setup_dataloader(dataset, args, mode='test')
            model.eval()
            cnt_accum = 0
            loss_accum = 0
            top5_acc_accum = 0
            top1_acc_accum = 0
            top5_acc_accum_easy = 0
            top1_acc_accum_easy = 0
            with torch.no_grad():
                for (inp, target, bbox) in val_dataloader:
                    if args.f16:
                        inp = inp.half()
                        target = target.half()
                    if args.cuda:
                        inp = inp.cuda()
                        target = target.cuda()
                    out = model(inp)
                    loss = grasp_loss(out, target)
                    loss_accum += loss.item() * len(inp)
                    bbox_pred, _, bbox_conf = feature2bboxwdeg(out.detach().cpu().numpy(), -np.inf)
                    bbox = bbox.numpy()
                    for aa, bb, cc in zip(bbox_pred, bbox, bbox_conf):
                        cc_order = np.argsort(-cc)
                        top5_acc_accum += (bbox_correct(aa[cc_order[:5]],bb,iou_t=cfg.iou_threshold,deg_t=cfg.deg_threshold)>0)
                        top1_acc_accum += bbox_correct(aa[cc_order[:1]],bb,iou_t=cfg.iou_threshold,deg_t=cfg.deg_threshold)
                        top5_acc_accum_easy += (bbox_correct(aa[cc_order[:5]],bb,iou_t=cfg.iou_threshold_easy,deg_t=cfg.deg_threshold_easy)>0)
                        top1_acc_accum_easy += bbox_correct(aa[cc_order[:1]],bb,iou_t=cfg.iou_threshold_easy,deg_t=cfg.deg_threshold_easy)
                    cnt_accum += len(inp)
                avg_val_loss = loss_accum / cnt_accum
                avg_val_top5_acc = top5_acc_accum / cnt_accum
                avg_val_top1_acc = top1_acc_accum / cnt_accum
                avg_val_top5_acc_easy = top5_acc_accum_easy / cnt_accum
                avg_val_top1_acc_easy = top1_acc_accum_easy / cnt_accum
            del val_dataloader

            if avg_val_top1_acc_easy>best_model_top1_easy: # improvement
                if len(args.save_path)>0: # save checkpoint
                    best_model_write_pth = args.save_path+"/w-f%02d.pth"%(fold_id+1)
                    sys.stderr.write("Top-1(easy) accuracy improved from %.4f to %.4f. Saving model to %s...\n"%(best_model_top1_easy, avg_val_top1_acc_easy, best_model_write_pth))
                    state_dict = base_model.state_dict()
                    state_dict['top1'] = avg_val_top1_acc
                    state_dict['top5'] = avg_val_top5_acc
                    state_dict['top1_easy'] = avg_val_top1_acc_easy
                    state_dict['top5_easy'] = avg_val_top5_acc_easy
                    torch.save(state_dict, best_model_write_pth)
                best_top1_acc = avg_val_top1_acc
                best_model_top5 = avg_val_top5_acc
                best_model_top1_easy = avg_val_top1_acc_easy
                best_model_top5_easy = avg_val_top5_acc_easy
            end_ts = time.time()
            elapsed_sec = end_ts-begin_ts
            elapsed = datetime.timedelta(seconds=int(elapsed_sec))
            nepochs_passed = args.nepoch*fold_id+e+1
            nepochs_remain = args.nepoch*cfg.n_folds - nepochs_passed
            eta = datetime.timedelta(seconds=int(elapsed_sec/nepochs_passed*nepochs_remain))
            logger.add_scalar('Fold-%d/loss'%(fold_id+1), avg_train_loss, e+1)
            logger.add_scalar('Fold-%d/val_loss'%(fold_id+1), avg_val_loss, e+1)
            logger.add_scalar('Fold-%d/top1-easy'%(fold_id+1), avg_train_top1_acc_easy, e+1)
            logger.add_scalar('Fold-%d/val_top1-easy'%(fold_id+1), avg_val_top1_acc_easy, e+1)
            logger.add_scalar('Fold-%d/top1'%(fold_id+1), avg_train_top1_acc, e+1)
            logger.add_scalar('Fold-%d/val_top1'%(fold_id+1), avg_val_top1_acc, e+1)
            logger.add_scalar('Fold-%d/top5-easy'%(fold_id+1), avg_train_top5_acc_easy, e+1)
            logger.add_scalar('Fold-%d/val_top5-easy'%(fold_id+1), avg_val_top5_acc_easy, e+1)
            logger.add_scalar('Fold-%d/top5'%(fold_id+1), avg_train_top5_acc, e+1)
            logger.add_scalar('Fold-%d/val_top5'%(fold_id+1), avg_val_top5_acc, e+1)
            print("F: [%2d/%2d] E: [%2d/%2d] l: %.4f vl: %.4f top5: %.2f(%.2f) top1: %.2f(%.2f) top5_easy: %.2f(%.2f) top1_easy: %.2f(%.2f) eta: %s"%(fold_id+1, cfg.n_folds, e+1, args.nepoch, avg_train_loss, avg_val_loss, avg_val_top5_acc, best_model_top5, avg_val_top1_acc, best_top1_acc, avg_val_top5_acc_easy, best_model_top5_easy, avg_val_top1_acc_easy, best_model_top1_easy, str(eta)))
            sys.stdout.flush()
        del model, base_model
        gc.collect()
        torch.cuda.empty_cache() # clear memory after every fold
        fold_top1[fold_id] = best_top1_acc
        fold_top5[fold_id] = best_model_top5
        fold_top1_easy[fold_id] = best_model_top1_easy
        fold_top5_easy[fold_id] = best_model_top5_easy
        print("Fold: [%2d/%2d] TOP-1: %.4f(%.4f) TOP-5: %.4f(%.4f)"%(fold_id+1, cfg.n_folds, best_top1_acc, best_model_top1_easy, best_model_top5, best_model_top5_easy))
        sys.stdout.flush()
    print("TOP-1(HARD): %.4f, STD: %.4f"%( np.mean(fold_top1), np.std(fold_top1)  ))
    print("TOP-5(HARD): %.4f, STD: %.4f"%( np.mean(fold_top5), np.std(fold_top5)  ))
    print("TOP-1(EASY): %.4f, STD: %.4f"%( np.mean(fold_top1_easy), np.std(fold_top1_easy)  ))
    print("TOP-5(EASY): %.4f, STD: %.4f"%( np.mean(fold_top5_easy), np.std(fold_top5_easy)  ))
    sys.stdout.flush()
    logger.close()

if __name__=='__main__':
    import torch.multiprocessing as mp
    #mp.set_start_method('forkserver', force=True)
    args = parse_args()
    print(args)
    main(args)

