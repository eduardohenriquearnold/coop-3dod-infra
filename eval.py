import os
import gc
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

from lib.dataset.coop_dataset import CooperativeDataset, DataLoader, Transform
from lib.models.voxelnet import Voxelnet
from lib.functions import load_helper
from lib.functions.nms import nms
from lib.evaluator import metrics


def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg


def build_data_loader(cfg, args):
    ext = torch.FloatTensor(cfg['shared']['area_extents']).view(3,2)
    ref = cfg['shared']['reference_loc']
    voxsize = cfg['shared']['voxel_size']
    maxpts = cfg['shared']['number_T']
    test_path = cfg['shared']['test_data']
    test_dataset = CooperativeDataset(test_path, ref, ext, voxsize, maxpts, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=False, pin_memory=False)
    return test_dataset, test_loader


@torch.no_grad()
def detectForCam(cfg, loader, model):

    score_threshold = cfg['test_rpn_proposal_cfg']['score_threshold']

    detections = []
    gts = []

    for iter, _input in tqdm(enumerate(loader), total=len(loader.dataset)//loader.batch_size, leave=False):
        gt_boxes = _input[9]
        voxel_with_points = _input[6]
        batch_size = voxel_with_points.shape[0]

        x = {
            'cfg': cfg,
            'image': _input[0],
            'points': _input[1],
            'indices': _input[2],
            'num_pts': _input[3],
            'leaf_out': _input[4],
            'voxel_indices': _input[5],
            'voxel_points': torch.autograd.Variable(_input[6]).cuda(),
            'ground_plane': _input[7],
            'gt_bboxes_2d': _input[8],
            'gt_bboxes_3d': _input[9],
            'num_divisions': _input[11]
        }

        outputs = model(x)
        outputs = outputs['predict']
        proposals = outputs[0].data.cpu().numpy()

        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()

        b_ix = 0
        rois_per_points_cloud = proposals[proposals[:, 0] == b_ix]
        if gt_boxes.shape[0] != 0:
            gts_per_points_cloud = gt_boxes[b_ix]
            gts_per_points_cloud = gts_per_points_cloud[gts_per_points_cloud[:,3] > 0] #Filter empty boxes (from batch)

            #Filter predictions by score 
            score_filter = rois_per_points_cloud[:, -1] > score_threshold
            filteredPred = rois_per_points_cloud[score_filter, 1:]

            if gts_per_points_cloud.shape[0] == 0:
                continue

            if filteredPred.shape[0] == 0:
                filteredPred = np.zeros((1,8))

            #accumulate metrics
            detections.append(filteredPred)
            gts.append(gts_per_points_cloud)
            
    gc.collect()
    return detections, gts

def main(args):
    cfg = load_config(args.config)

    test_dataset, test_loader = build_data_loader(cfg, args)

    # load model checkpoint
    device = torch.device("cuda:0")
    model = Voxelnet(cfg=cfg)
    assert os.path.isfile(args.checkpoint), '{} is not a valid file'.format(args.checkpoint)
    model = load_helper.load_checkpoint(model, args.checkpoint)
    model.cuda()
    model.eval()

    # available cameras
    cameras = args.cameras if len(args.cameras) > 0 else list(range(len(test_dataset.cameras)))

    print('Started late fusion predictions')
    detectionsLate = []
    gts = []
    for cam in cameras:
        #Set camera to use
        test_dataset.selectedCameras = [cam]

        #Get predictions and gt
        dets, gts = detectForCam(cfg, test_loader, model)
        detectionsLate.append(dets)
        print(f'Finished computing late-fusion detections for cam {cam}')


    print('Started early fusion predictions')
    test_dataset.selectedCameras = cameras
    detectionsEarly, gts = detectForCam(cfg, test_loader, model)
    print('Finished early fusion predictions')

    print('Started hybrid fusion predictions')
    test_dataset.selectedCameras = cameras
    test_dataset.R = cfg['shared']['hybrid_radius']
    detectionsHybrid, gts = detectForCam(cfg, test_loader, model)
    print('Finished hybrid fusion predictions')


    print(f'Detection Metrics for cameras {cameras}')
    for iou_threshold in [0.7, 0.8, 0.9]:
        evaluatorEarly = metrics.MetricsCalculator(iou_threshold=iou_threshold, bv=False)
        evaluatorLate = metrics.MetricsCalculator(iou_threshold=iou_threshold, bv=False)
        evaluatorHybrid = metrics.MetricsCalculator(iou_threshold=iou_threshold, bv=False)
        for i, gt in enumerate(gts):
            #late fusion of predictions from all cams
            predsLate= [detectionsLate[n][i] for n, _ in enumerate(cameras)]
            predsLate= np.concatenate(predsLate)
            predsLate = nms(predsLate, 0.1, 0.01)

            #Preds early
            predsEarly = detectionsEarly[i]
            predsEarly = nms(predsEarly, 0.1, 0.01)

            #Preds hybrid (late fusion + early fusion with radius R)
            predsHybrid = [detectionsLate[n][i] for n, _ in enumerate(cameras)] + [detectionsHybrid[i]] 
            predsHybrid = np.concatenate(predsHybrid)
            predsHybrid = nms(predsHybrid, 0.1, 0.01)

            #accumulate stats
            evaluatorLate.accumulate(predsLate, gt)
            evaluatorEarly.accumulate(predsEarly, gt)
            evaluatorHybrid.accumulate(predsHybrid, gt)

        #print results
        precL, recL = evaluatorLate.PR()
        precE, recE = evaluatorEarly.PR()
        precH, recH = evaluatorHybrid.PR()
        apL = evaluatorLate.AP(precL, recL)
        apE = evaluatorEarly.AP(precE, recE)
        apH = evaluatorHybrid.AP(precH, recH)
        print(f'Test AP w/IoU {iou_threshold}: Late  fusion: {apL.cpu().item():.4f}')
        print(f'Test AP w/IoU {iou_threshold}: Hybrid fusion: {apH.cpu().item():.4f}')
        print(f'Test AP w/IoU {iou_threshold}: Early fusion: {apE.cpu().item():.4f}')

    return 0 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True,
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True ,
                        help='path to model checkpoint')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--cameras', type=str, default='',
            help='string containing the ids of cameras to be used in the evaluation. E.g. 037 fuses data from cameras 0,3,7. Default: all available sensors are used')
    args = parser.parse_args()
    
    try:
        args.cameras = [int(c) for c in args.cameras]
    except ValueError:
        print('Invalid cameras id. Please provide a string of numbers (without spaces) indicating the cameras ids.')
    else:
        main(args)
