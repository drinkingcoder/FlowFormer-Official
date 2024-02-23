from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import evaluate_FlowFormer as evaluate
import evaluate_FlowFormer_tile as evaluate_tile
import core.datasets as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import vpd_utils


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def train(gpu, cfg):

    cfg.rank = gpu
    vpd_utils.setup_for_distributed(cfg.rank == 0)
    print("gpu = ",gpu)

    # coordinate multiple GPUs
    setup_ddp(gpu, cfg)
    rng = np.random.default_rng(12345)

    model = build_flowformer(cfg)
    device = torch.device(gpu)
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))
    
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.to(device)
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    # train_loader = datasets.fetch_dataloader(cfg)
    db = datasets.fetch_dataset(cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=cfg.world_size, rank=cfg.rank)

    train_loader = DataLoader(db, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=2)


    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    if cfg.rank == 0:
        logger = Logger(model, scheduler, cfg)

    add_noise = False
    if cfg.log_in_wandb and cfg.rank==0:
        wandb.init(project='1. Improving the ConvGRU of RAFT', name ="")


    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            flow_predictions = model(image1, image2, output)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            # logger.push(metrics)
            curr_epoch = total_steps // len(train_loader) + 1
            if cfg.rank == 0:
                logger.push(metrics, model, get_lr(optimizer), loss.item(), curr_epoch)


            ### change evaluate to functions

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                if cfg.rank == 0:
                    print("Doing validation: ")
                    PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                    torch.save(model.state_dict(), PATH)
                all_results={}

                results = {}
                # for val_dataset in args.validation:
                if cfg.validation == 'chairs':
                    results.update(evaluate.validate_chairs(model.module, args=cfg))
                elif cfg.validation == 'sintel':
                    results.update(evaluate.validate_sintel(model.module, args=cfg))
                elif cfg.validation == 'kitti':
                    results.update(evaluate.validate_kitti(model.module))
                
                all_results[cfg.validation] = results
                if cfg.rank == 0:
                    logger.write_dict(results, cfg.validation,cfg)

                # logger.write_dict(results)
                
                model.train()
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break
    
    if cfg.rank == 0:                
        logger.close()

        PATH = cfg.log_dir + '/final'
        os.makedirs(PATH,exist_ok=True)
        torch.save(model.state_dict(), PATH)

        PATH = f'{cfg.log_dir}/checkpoints/{cfg.stage}.pth'
        os.makedirs(PATH,exist_ok=True)
        torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='chairs', help="name your experiment")
    parser.add_argument('--stage', default='chairs',  help="determines which dataset to use for training") 
    parser.add_argument('--validation', default='chairs', type=str, nargs='+')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--log_in_wandb', default="true", type=str2bool)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--port', type=str, default="29500")
    parser.add_argument('--rank', type=int)


    args = parser.parse_args()
    args.world_size = args.num_gpus

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    cfg.log_dir = "log_dir_negroni"
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    mp.spawn(train, nprocs=args.num_gpus, args=(cfg,))
    # train(cfg)
