# encoding: utf-8

import numpy as np
import warnings
# from extensions._cython_bbox import cython_bbox


def bbox_iou_overlaps(b1, b2):
    # return cython_bbox.bbox_overlaps(b1.astype(np.float32), b2.astype(np.float32))
    '''
    :argument
        b1,b2: [n, k], k>=4, x1,y1,x2,y2,...
    :returns
        intersection-over-union pair-wise.
    '''
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
    union_area2 = (union_area1 - inter_area)
    return inter_area / np.maximum(union_area2, 1)


def bbox_iof_overlaps(b1, b2):
    '''
    :argument
        b1,b2: [n, k], k>=4 with x1,y1,x2,y2,....
    :returns
        intersection-over-former-box pair-wise
    '''
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    # area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1[:, np.newaxis], 1)


def center_to_corner(boxes):
    '''
    :argument
        boxes: [N, 4] of center_x, center_y, w, h
    :returns
        boxes: [N, 4] of xmin, ymin, xmax, ymax
    '''
    xmin = boxes[:, 0] - boxes[:, 2] / 2.
    ymin = boxes[:, 1] - boxes[:, 3] / 2.
    xmax = boxes[:, 0] + boxes[:, 2] / 2.
    ymax = boxes[:, 1] + boxes[:, 3] / 2.
    return np.vstack([xmin, ymin, xmax, ymax]).transpose()


def corner_to_center(boxes):
    '''
        inverse of center_to_corner
    '''
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    return np.vstack([cx, cy, w, h]).transpose()


def compute_loc_targets(raw_bboxes, gt_bboxes):
    '''
    :argument
        raw_bboxes, gt_bboxes:[N, k] first dim must be equal
    :returns
        loc_targets:[N, 4]
    '''
    bb = corner_to_center(raw_bboxes)  # cx, cy, w, h
    gt = corner_to_center(gt_bboxes)
    assert (np.all(bb[:, 2] > 0))
    assert (np.all(bb[:, 3] > 0))
    trgt_ctr_x = (gt[:, 0] - bb[:, 0]) / bb[:, 2]
    trgt_ctr_y = (gt[:, 1] - bb[:, 1]) / bb[:, 3]
    trgt_w = np.log(gt[:, 2] / bb[:, 2])
    trgt_h = np.log(gt[:, 3] / bb[:, 3])
    return np.vstack([trgt_ctr_x, trgt_ctr_y, trgt_w, trgt_h]).transpose()

def compute_loc_targets_3d(raw_bboxes, gt_bboxes):
    '''
    :argument
        raw_bboxes, gt_bboxes:[N, k] first dim must be equal
        raw_bboxes in the format N * [x, y, z, l, w, h, ry]
        gt_bboxes in the format N * [x, y, z, l, w, h, ry]
    :returns
        loc_targets:[N, 7]
    '''
    assert raw_bboxes.shape[-1]==7 and gt_bboxes.shape[-1]==7;

    l_a = raw_bboxes[:, 3]
    w_a = raw_bboxes[:, 4]
    h_a = raw_bboxes[:, 5]
    l_g = gt_bboxes[:, 3]
    w_g = gt_bboxes[:, 4]
    h_g = gt_bboxes[:, 5]
    d_a = np.sqrt(l_a*l_a + w_a*w_a)
    trgt_ctr_x = (gt_bboxes[:, 0] - raw_bboxes[:, 0]) / d_a
    trgt_ctr_y = (gt_bboxes[:, 1] - raw_bboxes[:, 1]) / h_a
    trgt_ctr_z = (gt_bboxes[:, 2] - raw_bboxes[:, 2]) / d_a
    trgt_l = np.log(l_g / l_a)
    trgt_w = np.log(w_g / w_a)
    trgt_h = np.log(h_g / h_a)
    trgt_theta = gt_bboxes[:, 6] - raw_bboxes[:, 6]
    return np.vstack([trgt_ctr_x, trgt_ctr_y, trgt_ctr_z, trgt_l, trgt_w, trgt_h, trgt_theta]).transpose()

def compute_loc_bboxes(raw_bboxes, deltas):
    '''
    :argument
        raw_bboxes, delta:[N, k] first dim must be equal
    :returns
        bboxes:[N, 4]
    '''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        bb = corner_to_center(raw_bboxes)  # cx, cy, w, h
        dt_cx = deltas[:, 0] * bb[:, 2] + bb[:, 0]
        dt_cy = deltas[:, 1] * bb[:, 3] + bb[:, 1]
        dt_w = np.exp(deltas[:, 2]) * bb[:, 2]
        dt_h = np.exp(deltas[:, 3]) * bb[:, 3]
        dt = np.vstack([dt_cx, dt_cy, dt_w, dt_h]).transpose()
        return center_to_corner(dt)

def compute_loc_bboxes_3d(raw_bboxes, deltas):
    '''
        :argument
            raw_bboxes, delta:[N, k] first dim must be equal
        :returns
            bboxes:[N, 7]
        '''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        l_a = raw_bboxes[:, 3]
        w_a = raw_bboxes[:, 4]
        h_a = raw_bboxes[:, 5]
        d_a = np.sqrt(l_a * l_a + w_a * w_a)

        dt_cx = deltas[:, 0] * d_a + raw_bboxes[:, 0]
        dt_cy = deltas[:, 1] * h_a + raw_bboxes[:, 1]
        dt_cz = deltas[:, 2] * d_a + raw_bboxes[:, 2]
        dt_l = np.exp(deltas[:, 3]) * l_a
        dt_w = np.exp(deltas[:, 4]) * w_a
        dt_h = np.exp(deltas[:, 5]) * h_a
        dt_theta = deltas[:, 6] + raw_bboxes[:, 6]
        dt = np.vstack([dt_cx, dt_cy, dt_cz, dt_l, dt_w, dt_h, dt_theta]).transpose()
        return dt

def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], 0, w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, h - 1)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, w - 1)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, h - 1)
    return bbox


def compute_recall(box_pred, box_gt):
    n_gt = box_gt.shape[0]
    if box_pred.size == 0 or n_gt == 0:
        return 0, n_gt
    ov = bbox_iou_overlaps(box_gt, box_pred)
    max_ov = np.max(ov, axis=1)
    idx = np.where(max_ov > 0.5)[0]
    n_rc = idx.size
    return n_rc, n_gt


def test_bbox_iou_overlaps():
    b1 = np.array([
        [0, 0, 4, 4],
        [1, 2, 3, 5],
        [5, 5, 5, 5]
    ])
    b2 = np.array([
        [0, 0, 4, 4],
        [1, 2, 3, 5],
        [5, 5, 5, 5],
        [100, 100, 200, 200]
    ])
    overlaps = bbox_iou_overlaps(b1, b2)
    print(overlaps)


def test_bbox_iof_overlaps():
    b1 = np.array([
        [0, 0, 4, 4],
        [1, 2, 3, 5],
        [5, 5, 5, 5]
    ])
    b2 = np.array([
        [0, 0, 4, 4],
        [1, 2, 3, 5],
        [5, 5, 5, 5],
        [100, 100, 200, 200]
    ])
    overlaps = bbox_iof_overlaps(b1, b2)
    print(overlaps)


def test_corner_center():
    b1 = np.array([
        [0, 0, 4, 4],
        [1, 2, 3, 5],
        [5, 5, 5, 5]
    ])
    b2 = corner_to_center(b1)
    b3 = center_to_corner(b2)
    print(b1)
    print(b2)
    print(b3)


def test_loc_trans():
    b1 = np.array([
        [0, 0, 4, 4],
        [1, 2, 3, 5],
        [4, 4, 5, 5]
    ])
    tg = np.array([
        [1, 1, 5, 5],
        [0, 2, 4, 5],
        [4, 4, 5, 5]
    ])
    deltas = compute_loc_targets(b1, tg)
    print(deltas)
    pred = compute_loc_bboxes(b1, deltas)
    print(pred)


def test_clip_bbox():
    b1 = np.array([
        [0, 0, 9, 29],
        [1, 2, 19, 39],
        [4, 4, 59, 59]
    ])
    print(b1)
    b2 = clip_bbox(b1, (30, 35))
    print(b2)


if __name__ == '__main__':
    # test_corner_center()
    # test_loc_trans()
    test_clip_bbox()
