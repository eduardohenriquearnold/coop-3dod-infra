#encoding: utf-8
from lib.functions import bbox_helper
from lib.functions import box_3d_encoder
from lib.functions import anchor_projector
from lib.functions import evaluation
import numpy as np
import torch
from torch.autograd import Variable
import logging
logger = logging.getLogger('global')

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().numpy() if torch.is_tensor(x) else x

def compute_anchor_targets(feature_size, anchors_overplane, cfg, ground_truth_bboxes, image_info, ignore_regions = None):
    r'''
    :argument
        cfg.keys(): {
            'anchor_ratios', anchor_scales, anchor_stride,
            negative_iou_thresh, ignore_iou_thresh,positive_iou_thresh,
            positive_percent, rpn_batch_size
        }
        feature_size: IntTensor, [4]. i.e. batch, num_anchors * 7, height, width
        anchors_overplane: numpy.ndarray, [K*A, 7], K=height*width
        ground_truth_bboxes: FloatTensor, [batch, max_num_gt_bboxes, 7]
        image_info: FloatTensor, [batch, 3]
        ignore_regions: FloatTensor, [batch, max_num_ignore_regions, 7]
    :returns
        cls_targets: Variable, [batch, num_anchors * 1, height, width]
        loc_targets, loc_masks: Variable, [batch, num_anchors * 7, height, width]
    '''
    ground_truth_bboxes, image_info, ignore_regions = \
        map(to_np_array, [ground_truth_bboxes, image_info, ignore_regions])

    batch_size, num_anchors_7, featmap_h, featmap_w = feature_size
    num_anchors = num_anchors_7 // 7
    assert(num_anchors * 7 == num_anchors_7)
    # # [A, 7]
    area_extents = np.asarray(cfg['area_extents']).reshape(-1, 2)
    bev_extents = area_extents[[0, 2]]

    B = batch_size
    A = num_anchors
    K = featmap_h * featmap_w
    G = ground_truth_bboxes.shape[1]
    logger.debug('anchors shape:{}'.format(anchors_overplane.shape))
    logger.debug('batchsize: %d, num_anchors: %d, K: %d, anchor shape[-1]: %d'%(B, A, K, G))
    logger.debug('ground_truth_bboxes size: {}'.format(ground_truth_bboxes.shape))
    #logger.info("the number of gts is {}".format(G))
    labels = np.zeros([B, K*A], dtype=np.int64)
    if G != 0:
        rpn_iou_type = cfg['rpn_iou_type']
        if rpn_iou_type == '2d':
            anchors = box_3d_encoder.box_3d_to_anchor(anchors_overplane)
            ground_truth_bboxes = ground_truth_bboxes.reshape(B * G, -1)
            gt_anchors = box_3d_encoder.box_3d_to_anchor(ground_truth_bboxes, ortho_rotate=True)

            # Convert anchors to 2d iou format
            anchors_for_2d_iou, _ = np.asarray(anchor_projector.project_to_bev(
                anchors, bev_extents))

            gt_boxes_for_2d_iou, _ = anchor_projector.project_to_bev(
                gt_anchors, bev_extents)

            # compute overlaps between anchors and gt_bboxes within each batch
            # shape: [B, K*A, G]
            logger.debug('gt_boxes_for_2d_iou size: {}'.format(gt_boxes_for_2d_iou.shape))
            gt_boxes_for_2d_iou = gt_boxes_for_2d_iou.reshape(B, G, -1)
            overlaps = np.stack([bbox_helper.bbox_iou_overlaps(anchors_for_2d_iou,
                                                               gt_boxes_for_2d_iou[ix]) for ix in range(B)], axis=0)
            logger.debug('overlaps shape:{}'.format(overlaps.shape))

        elif rpn_iou_type == '3d':
            ground_truth_bboxes = ground_truth_bboxes.reshape(B*G, -1)
            # Convert anchors to 3d iou format for calculation
            anchors_for_3d_iou = box_3d_encoder.box_3d_to_3d_iou_format(
                anchors_overplane)

            gt_boxes_for_3d_iou = box_3d_encoder.box_3d_to_3d_iou_format(ground_truth_bboxes)
            overlaps = np.zeros((B, anchors_overplane.shape[0], G))
            for b_ix in range(B):
                for i, gt_box in enumerate(gt_boxes_for_3d_iou[b_ix*G:(b_ix+1)*G]):
                    if np.any(gt_box > 0):
                        iou = evaluation.three_d_iou(gt_box, anchors_for_3d_iou)
                    else:
                        iou = np.zeros(anchors_overplane.shape[0])
                    overlaps[b_ix, :, i] = iou

            logger.debug('overlaps shape:{}'.format(overlaps.shape))

        elif rpn_iou_type == '2.5d':
            ground_truth_bboxes = ground_truth_bboxes.reshape(B * G, -1)
            # Convert anchors to 3d iou format for calculation
            anchors_for_3d_iou = box_3d_encoder.box_3d_to_3d_iou_format(
                anchors_overplane)

            gt_boxes_for_3d_iou = box_3d_encoder.box_3d_to_3d_iou_format(ground_truth_bboxes)
            overlaps = np.zeros((B, anchors_overplane.shape[0], G))
            for b_ix in range(B):
                for i, gt_box in enumerate(gt_boxes_for_3d_iou[b_ix * G:(b_ix + 1) * G]):
                    if np.any(gt_box > 0):
                        iou = evaluation.two_half_d_iou(gt_box, anchors_for_3d_iou)
                    else:
                        iou = np.zeros(anchors_overplane.shape[0])
                    overlaps[b_ix, :, i] = iou

        else:
            raise ValueError('Invalid rpn_iou_type {}', rpn_iou_type)

        ground_truth_bboxes = ground_truth_bboxes.reshape(B, G, -1)

        # overlaps shape: [B K*A G]
        # argmax_overlaps shape of [B, K*A]
        argmax_overlaps = overlaps.argmax(axis = 2)
        max_overlaps = overlaps.max(axis = 2)

        # [B, G]
        gt_max_overlaps = overlaps.max(axis=1)
        # ignore thoese gt_max_overlap too small
        gt_max_overlaps[gt_max_overlaps < 0.1] = -1
        gt_argmax_b_ix, gt_argmax_ka_ix, gt_argmax_g_ix = \
            np.where(overlaps == gt_max_overlaps[:, np.newaxis, :])
        # match each anchor to the ground truth bbox
        argmax_overlaps[gt_argmax_b_ix, gt_argmax_ka_ix] = gt_argmax_g_ix

        labels[max_overlaps < cfg['negative_iou_thresh']] = 0

        # remove negatives located in ignore regions
        if ignore_regions is not None:
            #logger.info('Anchor Ignore')
            iof_overlaps = np.stack([bbox_helper.bbox_iof_overlaps
                                         (anchors_overplane, ignore_regions[ix])
                                     for ix in range(B)], axis=0)
            max_iof_overlaps = iof_overlaps.max(axis=2)  # [B, K*A]
            labels[max_iof_overlaps > cfg['ignore_iou_thresh']] = -1

        labels[gt_argmax_b_ix, gt_argmax_ka_ix] = 1
        labels[max_overlaps > cfg['positive_iou_thresh']] = 1
    # sampling
    num_pos_sampling = int(cfg['positive_percent'] * cfg['rpn_batch_size'] * batch_size)
    pos_b_ix, pos_ka_ix = np.where(labels > 0)
    num_positives = len(pos_b_ix)
    print("positive :", num_positives)
    if num_positives > num_pos_sampling:
        remove_ix = np.random.choice(num_positives, size = num_positives - num_pos_sampling, replace = False)
        labels[pos_b_ix[remove_ix], pos_ka_ix[remove_ix]] = -1
        num_positives = num_pos_sampling
    num_neg_sampling = cfg['rpn_batch_size'] * batch_size - num_positives
    neg_b_ix, neg_ka_ix = np.where(labels == 0)
    num_negatives = len(neg_b_ix)
    if num_negatives > num_neg_sampling:
        remove_ix = np.random.choice(num_negatives, size = num_negatives - num_neg_sampling, replace = False)
        labels[neg_b_ix[remove_ix], neg_ka_ix[remove_ix]] = -1

    loc_targets = np.zeros([B, K*A, 7], dtype=np.float32)
    loc_masks = np.zeros([B, K*A, 7], dtype=np.float32)
    if G != 0:
        pos_b_ix, pos_ka_ix = np.where(labels > 0)
        pos_anchors = anchors_overplane[pos_ka_ix, :]

        pos_target_ix = argmax_overlaps[pos_b_ix, pos_ka_ix]
        pos_target_gt = ground_truth_bboxes[pos_b_ix, pos_target_ix]
        pos_loc_targets = bbox_helper.compute_loc_targets_3d(pos_anchors, pos_target_gt)

        loc_targets[pos_b_ix, pos_ka_ix, :] = pos_loc_targets
        # loc_weights = np.zeros([B, K*A, 4])
        loc_masks[pos_b_ix, pos_ka_ix, :] = 1.

    # transpose to match the predicted convolution shape
    cls_targets = Variable(
        torch.from_numpy(labels).long().view(B, featmap_h, featmap_w, A).permute(0, 3, 1, 2)).cuda().contiguous()
    loc_targets = Variable(
        torch.from_numpy(loc_targets).float().view(B, featmap_h, featmap_w, A * 7).permute(0, 3, 1, 2)).cuda().contiguous()
    loc_masks = Variable(
        torch.from_numpy(loc_masks).float().view(B, featmap_h, featmap_w, A * 7).permute(0, 3, 1, 2)).cuda().contiguous()
    loc_nomalizer = max(1,len(np.where(labels > 0)[0]))
    logger.debug('positive anchors:%d' % len(pos_b_ix))
    return cls_targets, loc_targets, loc_masks, loc_nomalizer
