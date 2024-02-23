import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.default import get_cfg
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
import datasets
from core.utils import flow_viz
from core.utils import frame_utils

# from FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from core.raft import RAFT
from core.datasets import FlyingChairs, MpiSintel


from core.utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm
import vpd_utils

@torch.no_grad()
def validate_chairs(model, args=None):
    """ Perform evaluation on the FlyingChairs (test) split """
    args.val_workers=2
    model.eval()

    epe_list = []
    ddp_logger = vpd_utils.MetricLogger()

    val_dataset = FlyingChairs(split='validation')
    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val, num_workers=args.val_workers, pin_memory=True)
    device = torch.device(args.rank)

    # for val_id in range(len(val_dataset)):
    for batch in tqdm(val_loader):
        image1, image2, flow_gt, _ = batch
        image1 = image1.to(device)
        image2 = image2.to(device)
        flow_pre, _ = model(image1, image2)

        epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    # epe = np.mean(np.concatenate(epe_list))
    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)

    if args.rank == 0:
        print("Validation:- EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, px1, px3, px5))

    ddp_logger.update(epe=float(epe))
    ddp_logger.synchronize_between_processes()
    epe = ddp_logger.meters['epe'].global_avg

    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}

@torch.no_grad()
def validate_sintel(model, args=None):

    """ Peform validation using the Sintel (train) split """
    args.val_workers=2
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        print("Validating on %s" % dstype)
        val_dataset = MpiSintel(split='training', dstype=dstype)


        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val, num_workers=args.val_workers, pin_memory=True)
        device = torch.device(args.rank)
        
        epe_list = []
        ddp_logger = vpd_utils.MetricLogger()

        for batch in tqdm(val_loader):
        # for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = batch
            image1 = image1.to(device)
            image2 = image2.to(device)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pr = model(image1, image2)
            flow = padder.unpad(flow_pr[0]).cpu()[0]

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        ddp_logger.update(epe=float(epe))
        ddp_logger.update(px1=float(px1))
        ddp_logger.update(px3=float(px3))
        ddp_logger.update(px5=float(px5))
        ddp_logger.synchronize_between_processes()
        epe = ddp_logger.meters['epe'].global_avg
        px1 = ddp_logger.meters['px1'].global_avg
        px3 = ddp_logger.meters['px3'].global_avg
        px5 = ddp_logger.meters['px5'].global_avg

        if args.rank == 0:
            print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        # import ipdb;ipdb.set_trace()
        # results[dstype] = np.mean(epe_list)
        results[dstype+"_epe"] = epe
        results[dstype+"_1px"] = px1
        results[dstype+"_3px"] = px3
        results[dstype+"_5px"] = px5

    return results

@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """

    model.eval()
    for dstype in ['final', "clean"]:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre = model(image1, image2)

            flow_pre = padder.unpad(flow_pre[0]).cpu()
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)


@torch.no_grad()
def validate_kitti(model):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre = model(image1, image2)

        flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

        epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # cfg = get_cfg()
    if args.small:
        cfg = get_small_things_cfg()
    else:
        cfg = get_things_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    print(args)

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'sintel_submission':
            create_sintel_submission(model.module)


