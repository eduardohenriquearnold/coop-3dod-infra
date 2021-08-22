import torch.nn.functional as F
import torch.nn as nn
import torch
import functools
import logging
import numpy as np

from lib.functions import anchor_target_3d
from lib.functions import rpn_proposal
from lib.functions import anchor_helper
logger = logging.getLogger('global')

class model(nn.Module):
    def __init__(self, cfg):
        super(model, self).__init__()

    def feature_extractor(self, voxel_with_points, num_pts, leaf_out, voxel_indices, num_divisions):
        raise NotImplementedError

    def rpn(self, x):
        raise NotImplementedError

    def rcnn(self, x, rois):
        pass

    def generate_anchors(self, feature_size, cfg, ground_plane=None):
        """
        generate anchors.
        :param feature_size:  batch_size, num_anchors*7, featmap_h, featmap_w
        :param cfg: include area_extents, anchor_3d_sizes, ground_plane, anchor_stride
        :param ground_plane: Fixed or read from a file
        :return: anchors, shape[A*K, 7],everyone is like [x,y,z,l,w,h,ry]
        """

        batch_size, num_anchors_7, featmap_h, featmap_w = feature_size
        num_anchors = num_anchors_7 // 7
        # [A, 7]
        area_extents = np.asarray(cfg['area_extents']).reshape(-1, 2)
        anchor_3d_sizes = np.asarray(cfg['anchor_3d_sizes']).reshape(-1, 3)
        # ground_plane = np.asarray(cfg['ground_plane'])
        anchor_stride = np.asarray(cfg['anchor_stride'])
        anchors_overplane = anchor_helper.get_anchors_over_plane(featmap_h, featmap_w, area_extents, anchor_3d_sizes,
                                                                 anchor_stride, ground_plane)
        return anchors_overplane

    def _add_rpn_loss(self, compute_anchor_targets_fn, rpn_pred_cls,
                      rpn_pred_loc, anchors):
        '''
        :param compute_anchor_targets_fn: functions to produce anchors' learning targets.
        :param rpn_pred_cls: [B, num_anchors * 2, h, w], output of rpn for classification.
        :param rpn_pred_loc: [B, num_anchors * 7, h, w], output of rpn for localization.
        :return: loss of classification and localization, respectively.
        '''
        # [B, num_anchors * 2, h, w], [B, num_anchors * 7, h, w]
        cls_targets, loc_targets, loc_masks, loc_normalizer = \
                compute_anchor_targets_fn(rpn_pred_loc.size(), anchors)

        # tranpose to the input format of softmax_loss function
        rpn_pred_cls = rpn_pred_cls.permute(0,2,3,1).contiguous().view(-1, 2)
        cls_targets = cls_targets.permute(0,2,3,1).contiguous().view(-1)
        rpn_loss_cls = F.cross_entropy(
            rpn_pred_cls, cls_targets, ignore_index=-1)
        # mask out negative anchors
        rpn_loss_loc = smooth_l1_loss_with_sigma(rpn_pred_loc * loc_masks,
                                                 loc_targets) / loc_normalizer

        # classification accuracy, top1
        acc = accuracy(rpn_pred_cls.data, cls_targets.data)[0]
        return rpn_loss_cls, rpn_loss_loc, acc

    def _pin_args_to_fn(self, cfg, ground_truth_bboxes, image_info, ignore_regions):
        partial_fn = {}
        if self.training:
            partial_fn['anchor_target_fn'] = functools.partial(
                anchor_target_3d.compute_anchor_targets,
                cfg=cfg['train_anchor_target_cfg'],
                ground_truth_bboxes=ground_truth_bboxes,
                ignore_regions=ignore_regions,
                image_info=image_info)
            partial_fn['rpn_proposal_fn'] = functools.partial(
                rpn_proposal.compute_rpn_proposals,
                cfg=cfg['train_rpn_proposal_cfg'],
                image_info=image_info)
        else:
            partial_fn['rpn_proposal_fn'] = functools.partial(
                rpn_proposal.compute_rpn_proposals,
                cfg=cfg['test_rpn_proposal_cfg'],
                image_info=image_info)

        return partial_fn

    def forward(self, input):
        cfg = input['cfg']
        # images = input['image']
        points = input['points']
        indices = input['indices']
        num_pts = input['num_pts']
        leaf_out = input['leaf_out']
        voxel_indices = input['voxel_indices']
        voxel_with_points = input['voxel_points']
        gt_bboxes_2d = input['gt_bboxes_2d']
        gt_bboxes_3d = input['gt_bboxes_3d']
        ground_plane = input['ground_plane']
        num_divisions = input['num_divisions']

        partial_fn = self._pin_args_to_fn(
                cfg,
                gt_bboxes_3d,
                image_info=None,
                ignore_regions=None)

        outputs = {'losses': [], 'predict': [], 'accuracy': []}
        x = self.feature_extractor(voxel_with_points, num_pts, leaf_out, voxel_indices, num_divisions)
        rpn_pred_cls, rpn_pred_loc = self.rpn(x)
        logger.debug("rpn_pred_cls shape: {}".format(rpn_pred_cls.size()))
        logger.debug("rpn_pred_loc shape: {}".format(rpn_pred_loc.size()))

        anchors = self.generate_anchors(rpn_pred_loc.size(), cfg['shared'])

        if self.training:
            # train rpn
            rpn_loss_cls, rpn_loss_loc, rpn_acc = \
                    self._add_rpn_loss(partial_fn['anchor_target_fn'],
                            rpn_pred_cls, rpn_pred_loc, anchors)

            # get rpn proposals
            #compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            #rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
            #rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
            #rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)
            #proposals = compute_rpn_proposals_fn(rpn_pred_cls.data, rpn_pred_loc.data, anchors)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc]
            outputs['accuracy'] = [rpn_acc]
            #outputs['predict'] = [proposals]
        else:
            # rpn test
            compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
            rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
            rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)
            proposals = compute_rpn_proposals_fn(rpn_pred_cls.data, rpn_pred_loc.data, anchors)
            
            outputs['predict'] = [proposals, anchors]
        return outputs

def smooth_l1_loss_with_sigma(pred, targets, sigma=3.0):
    sigma_2 = sigma**2
    diff = pred - targets
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    loss = torch.pow(diff, 2) * sigma_2 / 2. * smoothL1_sign \
            + (abs_diff - 0.5 / sigma_2) * (1. - smoothL1_sign)
    reduced_loss = torch.sum(loss)
    return reduced_loss

def accuracy(output, target, topk=(1, ), ignore_index=-1):
    """Computes the precision@k for the specified values of k"""
    keep = torch.nonzero(target != ignore_index).squeeze()
    #logger.info('target.shape:{0}, keep.shape:{1}'.format(target.shape, keep.shape))
    assert (keep.dim() == 1)
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
