"""
This module converts data to and from the 'box_3d' format
 [x, y, z, l, w, h, ry]
"""
import numpy as np


def box_3d_to_anchor(boxes_3d, ortho_rotate=False):
    """ Converts a box_3d [x, y, z, l, w, h, ry]
    into anchor form [x, y, z, dim_x, dim_y, dim_z]

    Anchors in box_3d format should have an ry of 0 or 90 degrees.
    l and w will be matched to dim_x or dim_z depending on the rotation,
    while h will always correspond to dim_y

    Args:
        boxes_3d: N x 7 ndarray of box_3d
        ortho_rotate: optional, if True the box is rotated to the
            nearest 90 degree angle, or else the box is projected
            onto the x and z axes

    Returns:
        N x 6 ndarray of anchors in 'anchor' form
    """

    boxes_3d = np.asarray(boxes_3d).reshape(-1, 7)

    # fc.check_box_3d_format(boxes_3d)

    num_anchors = len(boxes_3d)
    anchors = np.zeros((num_anchors, 6))

    # Set x, y, z
    anchors[:, [0, 1, 2]] = boxes_3d[:, [0, 1, 2]]

    # Dimensions along x, y, z
    box_l = boxes_3d[:, [3]]
    box_w = boxes_3d[:, [4]]
    box_h = boxes_3d[:, [5]]
    box_ry = boxes_3d[:, [6]]

    # Rotate to nearest multiple of 90 degrees
    if ortho_rotate:
        half_pi = np.pi / 2
        box_ry = np.round(box_ry / half_pi) * half_pi

    cos_ry = np.abs(np.cos(box_ry))
    sin_ry = np.abs(np.sin(box_ry))

    # dim_x, dim_y, dim_z
    anchors[:, [3]] = box_l * cos_ry + box_w * sin_ry
    anchors[:, [4]] = box_h
    anchors[:, [5]] = box_w * cos_ry + box_l * sin_ry

    return anchors

def box_3d_to_3d_iou_format(boxes_3d):
    """ Returns a numpy array of 3d box format for iou calculation
    Args:
        boxes_3d: list of 3d boxes
    Returns:
        new_anchor_list: numpy array of 3d box format for iou
    """
    boxes_3d = np.asarray(boxes_3d)
    # fc.check_box_3d_format(boxes_3d)

    iou_3d_boxes = np.zeros([len(boxes_3d), 7])
    iou_3d_boxes[:, 4:7] = boxes_3d[:, 0:3]
    iou_3d_boxes[:, 1] = boxes_3d[:, 3]
    iou_3d_boxes[:, 2] = boxes_3d[:, 4]
    iou_3d_boxes[:, 3] = boxes_3d[:, 5]
    iou_3d_boxes[:, 0] = boxes_3d[:, 6]

    return iou_3d_boxes
