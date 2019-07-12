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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from torch import distributed
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
        '--momentum', type=float, default=0.99, help='Momentum of SGD'
    )
    parser.add_argument(
        '--save_path', type=str, default='./weights', help=''
    )
    parser.add_argument(
        '--split_mode', type=str, default='obj', help='obj/img'
    )
    # vgg11/vgg16/vgg19/vgg11_bn/vgg16_bn/vgg19_bn/mobilenetv2/resnet18/resnet34/resnet50/resnet101
    parser.add_argument(
        '--backbone', type=str, default='vgg19_bn', help='Model backbone (vgg11/vgg16/vgg19/vgg11_bn/vgg16_bn/vgg19_bn/mobilenetv2/resnet18/resnet34/resnet50/resnet101)'
    )
    parser.add_argument(
        '--n_workers', type=int, default=6, help='Multiprocessing'
    )
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')
    parser.add_argument('--f16', action='store_true', help='Half precision training')
    parser.add_argument('--with_fc', action='store_true', help='With fc layers (like YOLOv1)')
    parser.add_argument('--backend',type=str,default='gloo',help='Name of the backend to use.')
    parser.add_argument('-i','--init-method',type=str,default='tcp://127.0.0.1:23456',help='URL specifying how to initialize the package.')
    parser.add_argument('-r', '--rank', type=int, help='Rank of the current process.', default=0)
    parser.add_argument('-s','--world-size',type=int,help='Number of processes participating in the job.', default=0)
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    return args

def setup_dataloader(dataset, sampler, args, mode='train'):
    dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.n_workers,
                pin_memory=True if args.cuda else False,
                drop_last=(mode=='train'),
                sampler=sampler
                )
    return dataloader

def main(args):
    if len(args.save_path)>0 and (not os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    if args.world_size>1:
        distributed.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            rank=args.rank,
            world_size=args.world_size
        )
    print('===***  Preprocessing ***===')
    sys.stdout.flush()
    dataset = CornellGraspDataset(split_mode=args.split_mode, return_box=True)
    print('Done!')
    sys.stdout.flush()

    begin_ts = time.time()
    fold_acc = np.zeros(cfg.n_folds)
    print('===*** Begin Training ***===')
    sys.stdout.flush()
    for fold_id in range(cfg.n_folds):
        base_model = GraspModel(backbone=args.backbone, with_fc=args.with_fc)
        if args.f16:
            base_model = base_model.half()
            for layer in base_model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
        if args.cuda:
            base_model = base_model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(base_model) if args.world_size>1 else base_model
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,20,90], gamma=0.2)

        ### Setup current fold ###
        dataset.set_current_fold(fold_id)

        best_val_acc = -np.inf
        best_loss = np.inf
        for e in range(args.nepoch):
            dataset.set_current_state('train')
            train_sampler = DistributedSampler(dataset) if args.world_size>1 else RandomSampler(dataset)
            if args.world_size>1:
                train_sampler.set_epoch(e)
            dataloader = setup_dataloader(dataset, train_sampler, args, mode='train')
            #assert sampler_ref is train_sampler
            if args.world_size>1:
                train_sampler.set_epoch(e)
            lr_scheduler.step()
            model.train()
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
                bbox_pred = feature2bboxwdeg(out.detach().cpu().numpy(), cfg.hinge_margin)[0]
                bbox = bbox.numpy()
                for aa, bb in zip(bbox_pred, bbox):
                    acc_accum += bbox_correct(aa,bb)
                    bbox_accum += len(aa)
                cnt_accum += len(inp)
            avg_train_loss = loss_accum / cnt_accum
            avg_train_acc = acc_accum / (bbox_accum+1e-8)
            del dataloader, train_sampler

            dataset.set_current_state('test')
            val_sampler = SequentialSampler(dataset)
            val_dataloader = setup_dataloader(dataset, val_sampler, args, mode='test')
            #assert sampler_ref is val_sampler
            model.eval()
            cnt_accum = 0
            bbox_accum = 0
            loss_accum = 0
            acc_accum = 0
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
                    bbox_pred = feature2bboxwdeg(out.detach().cpu().numpy(), cfg.hinge_margin)[0]
                    bbox = bbox.numpy()
                    for aa, bb in zip(bbox_pred, bbox):
                        acc_accum += bbox_correct(aa,bb)
                        bbox_accum += len(aa)
                    cnt_accum += len(inp)
                avg_val_loss = loss_accum / cnt_accum
                avg_val_acc = acc_accum / (bbox_accum+1e-8)
            del val_dataloader, val_sampler

            if avg_val_acc>best_val_acc or (abs(avg_val_acc-best_val_acc)<1e-5 and avg_val_loss<best_loss): # improvement
                if len(args.save_path)>0 and args.rank==0: # save checkpoint
                    best_model_write_pth = args.save_path+"/w-f%02d.pth"%(fold_id+1)
                    sys.stderr.write("Accuracy improved from %.4f to %.4f. Saving model to %s...\n"%(best_val_acc, avg_val_acc, best_model_write_pth))
                    state_dict = base_model.state_dict()
                    state_dict['val_accuracy'] = avg_val_acc
                    torch.save(state_dict, best_model_write_pth)
            best_val_acc = max(best_val_acc, avg_val_acc)
            best_loss = min(best_loss, avg_val_loss)
            end_ts = time.time()
            elapsed_sec = end_ts-begin_ts
            elapsed = datetime.timedelta(seconds=int(elapsed_sec))
            nepochs_passed = args.nepoch*fold_id+e+1
            nepochs_remain = args.nepoch*cfg.n_folds - nepochs_passed
            eta = datetime.timedelta(seconds=int(elapsed_sec/nepochs_passed*nepochs_remain))
            print("F: [%2d/%2d] E: [%2d/%2d] l: %.4f vl: %.4f a: %.4f va: %.4f(%.4f) ela: %s eta: %s"%(fold_id+1, cfg.n_folds, e+1, args.nepoch, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, best_val_acc, str(elapsed), str(eta)))
            sys.stdout.flush()
        del model, base_model
        gc.collect()
        torch.cuda.empty_cache() # clear memory after every fold
        fold_acc[fold_id] = best_val_acc
        print("Fold: [%2d/%2d] ACC: %.4f"%(fold_id+1, cfg.n_folds, best_val_acc))
        sys.stdout.flush()
    print("ACC: %.4f, STD: %.4f"%( np.mean(fold_acc), np.std(fold_acc)  ))
    sys.stdout.flush()

if __name__=='__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('forkserver', force=True)
    args = parse_args()
    print(args)
    main(args)

