#encoding: utf-8
from lib.functions import bbox_helper
from lib.extensions._nms.pth_nms import pth_nms
from lib.functions import box_3d_encoder
from lib.functions import anchor_projector
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
logger = logging.getLogger('global')

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().numpy() if torch.is_tensor(x) else x

def compute_rpn_proposals(conv_cls, conv_loc, anchors_overplane, cfg, image_info, ground_plane = None):
    '''
    :argument
        cfg: configs
        conv_cls: FloatTensor, [batch, num_anchors * 2, h, w], conv output of classification
        conv_loc: FloatTensor, [batch, num_anchors * 7, h, w], conv output of localization
        image_info: FloatTensor, [batch, 3], image size
    :returns
        proposals: Variable, [N, 5], 2-dim: batch_ix, x1, y1, x2, y2
    '''

    batch_size, num_anchors_7, featmap_h, featmap_w = conv_loc.shape
    # [K*A, 4]
    # anchors_overplane = anchor_helper.get_anchors_over_plane(featmap_h, featmap_w,
    #                                                          cfg['anchor_ratios'], cfg['anchor_scales'], cfg['anchor_stride'])
    # [A, 7]
    area_extents = np.asarray(cfg['area_extents']).reshape(-1, 2)
    bev_extents = area_extents[[0, 2]]

    B = batch_size
    A = num_anchors = num_anchors_7 // 7
    assert(A * 7 == num_anchors_7)
    K = featmap_h * featmap_w

    cls_view = conv_cls.permute(0, 2, 3, 1).contiguous().view(B, K*A, -1).cpu().numpy()
    loc_view = conv_loc.permute(0, 2, 3, 1).contiguous().view(B, K*A, 7).cpu().numpy()
    if torch.is_tensor(image_info):
        image_info = image_info.cpu().numpy()

    #all_proposals = [bbox_helper.compute_loc_bboxes(anchors_overplane, loc_view[ix]) for ix in range(B)]
    # [B, K*A, 4]
    #pred_loc = np.stack(all_proposals, axis = 0)
    #pred_cls = cls_view
    batch_proposals = []
    pre_nms_top_n = cfg['pre_nms_top_n']
    for b_ix in range(B):
        scores = cls_view[b_ix, :, -1] # to compatible with sigmoid
        if pre_nms_top_n <= 0 or pre_nms_top_n > scores.shape[0]:
            order = scores.argsort()[::-1]
        else:
            inds = np.argpartition(-scores, pre_nms_top_n)[:pre_nms_top_n]
            order = np.argsort(-scores[inds])
            order = inds[order]
        loc_delta = loc_view[b_ix, order, :]
        loc_anchors = anchors_overplane[order, :]
        scores = scores[order]
        boxes_3d = bbox_helper.compute_loc_bboxes_3d(loc_anchors, loc_delta)
        boxes_3d_anchor = box_3d_encoder.box_3d_to_anchor(boxes_3d)
        bev_boxes, _ = anchor_projector.project_to_bev(boxes_3d_anchor, bev_extents)
        logger.debug('bev_boxes shape: {}, scores shape: {}'.format(bev_boxes.shape, scores.shape))
        # boxes = bbox_helper.clip_bbox(boxes, image_info[b_ix])
        bev_proposals = np.hstack([bev_boxes, scores[:, np.newaxis]])
        bev_proposals = bev_proposals[(bev_proposals[:, 2] - bev_proposals[:, 0] + 1 >= cfg['roi_min_size'])
                            & (bev_proposals[:, 3] - bev_proposals[:, 1] + 1 >= cfg['roi_min_size'])]
        logger.debug('proposals has done.')
        keep_index = pth_nms(torch.from_numpy(bev_proposals).float().cuda(), cfg['nms_iou_thresh']).numpy()
        if cfg['post_nms_top_n'] > 0:
            keep_index = keep_index[:cfg['post_nms_top_n']]

        batch_ix = np.full(keep_index.shape, b_ix)
        proposals = np.hstack([batch_ix[:, np.newaxis], boxes_3d[keep_index, :], scores[keep_index, np.newaxis]])
        batch_proposals.append(proposals)
    batch_proposals = torch.autograd.Variable(torch.from_numpy(np.vstack(batch_proposals))).float()
    if batch_proposals.dim() < 2:
        batch_proposals.unsqueeze(dim=0)
    logger.debug('batch proposals shape: {}'.format(batch_proposals.size()))
    return batch_proposals
