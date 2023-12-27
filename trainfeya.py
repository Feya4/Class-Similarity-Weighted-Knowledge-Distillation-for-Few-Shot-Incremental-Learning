import os
import os
import sys
sys.path.append('./')
import tqdm
import math
import logging
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter 

from utils.D_utils import *
from dataloader.samplers import *
from methods.feyaClassifier import feyaClassifier

from utils.utils_inc import * 
from utils.f_inc import * 
from sync_batchnorm import convert_model
import psutil
import time

start_time = time.time()

# Code before model execution

# Track memory usage before model execution
memory_before = psutil.virtual_memory().used 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiments arguments
    parser.add_argument('--dataroot', type=str, default='/media/meng1/disk1/feidu/projects/FeyaFSCIL/data')
   # parser.add_argument('--dataset', type=str, default='mini-imagenet')
    parser.add_argument('--dataset', type=str, default='cifar100')
    #parser.add_argument('--dataset', type=str, default='cub200')
    parser.add_argument('--methods', type=str, default='imprint')
    parser.add_argument('--b_mode', type=str, default='avg_cos')
    parser.add_argument('--first_norm', action='store_true')
    parser.add_argument('--dir_exp', type=str, default='experiment')
    # training arguments
    parser.add_argument('--epoch', type=int, default=160)#120
    parser.add_argument('--batch_size', type=int, default=128)#128
    parser.add_argument('--new_batch_size', type=int, default=0)#0
    parser.add_argument('--batch_size_test', type=int, default=100)#100
    parser.add_argument('--lr_init', type=float, default=-1)
    parser.add_argument('--schedule', type=str, default='Milestone', choices=['Step', 'Milestone'])
    parser.add_argument('--milestones', nargs='+', type=int, default=-1)
    parser.add_argument('--step', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--val_start', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--change_val_interval', type=int, default=70)
    parser.add_argument('--num_workers', type=int, default=8)#8
    parser.add_argument('--binary_report', action='store_true')
    parser.add_argument('-gpu', default='0,1,2,3')
    args = parser.parse_args()
    args = dataset_up_sets(args)
    args.first_norm = True

    if args.lr_init == -1:
        if args.dataset == 'cifar100':
            args.lr_init = 0.1
        elif args.dataset == 'mini_imagenet':
            args.lr_init = 0.1
        elif args.dataset == 'cub200':
            args.lr_init = 0.01
        else:
            Exception('Undefined dataset name!')

    if args.milestones == -1:
        if args.dataset == 'cifar100':
            args.milestones = [120, 160]
        elif args.dataset == 'mini_imagenet':
            args.milestones = [120, 160]
        elif args.dataset == 'cub200':
            args.milestones = [50, 70, 90]
        else:
            Exception('Undefined dataset name!')

    args.checkpoint_dir = '%s/%s' %(args.dir_exp, args.dataset)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    logging.basicConfig(filename=os.path.join(args.checkpoint_dir, 'train.log'), level=logging.INFO)
    logging.info(args)

    print(args)

    # init model
    model = feyaClassifier(args, phase='pretrain')
    #model.cpu()
    #model.cuda()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print('parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # init optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.schedule == 'Step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    elif args.schedule == 'Milestone':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss()

    # dataset in pre-training phase
    train_set, train_loader, test_loader = model.getdataloader(0)
    print(len(train_set))
    
    # training
    best_test_acc_base    = 0
    best_test_epoch_base  = 0 
    for epoch in range(args.epoch):
        if epoch >= args.change_val_interval:
            args.val_interval = 1
        
        # torch.cuda.empty_cache()
        model.train()
        if args.schedule != 'Step' and args.schedule != 'Milestone':
            adjust_learning_rate(optimizer, epoch, lr_init=args.lr_init, n_epoch=args.epoch)
        
        tqdm_gen = tqdm.tqdm(train_loader)
        loss_avg = 0
        for i, X in enumerate(tqdm_gen): 
            data, label = X
            data = data.cpu()#CUDA
            label = label.cpu()#cuda()
            pred = model(flag='forward_base', input=data)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm_gen.set_description('e:%d loss = %.4f' % (epoch, loss.item()))
            loss_avg += loss.item()
        if args.schedule == 'Step' or args.schedule == 'Milestone':
            lr_scheduler.step()
        
        out_str = '======epoch: %d avg loss: %.6f======'%(epoch, loss_avg/len(train_loader))
        print(out_str)
        logging.info(out_str)
        
        end_time = time.time()
        running_time = end_time - start_time

        # Measure memory usage after model execution
        memory_after = psutil.virtual_memory().used

         # Print memory usage and running time
        print(f"Memory usage: {memory_after - memory_before} bytes")
        print(f"Running time: {running_time} seconds")
        # testing
        model.eval()
        if (epoch == 0) or (epoch > args.val_start and (epoch+1) % args.val_interval == 0):
            acc_list = model.inc_testloop(epoch=epoch)
            test_acc_base = acc_list[0]
            if test_acc_base > best_test_acc_base:
                best_test_acc_base = test_acc_base
                best_test_epoch_base = epoch
                outfile = os.path.join(args.checkpoint_dir, 'best_model_%s.tar'%(args.b_mode))
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    
    out_str = '==========Epoch: %d Best Base Test acc = %.2f%%==========='%(best_test_epoch_base, 100*best_test_acc_base)
    print(out_str)
    logging.info(out_str)
