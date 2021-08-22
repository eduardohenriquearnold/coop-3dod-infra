"""
Projects anchors into bird's eye view and image space.
Returns the minimum and maximum box corners, and will only work
for anchors rotated at 0 or 90 degrees
"""

import numpy as np
from lib.dataset.kitti_util import compute_numpy_boxes_3d
# import tensorflow as tf

# from wavedata.tools.core import calib_utils


def project_to_bev(anchors, bev_extents):
    """
    Projects an array of 3D anchors into bird's eye view

    Args:
        anchors: list of anchors in anchor format (N x 6):
            N x [x, y, z, dim_x, dim_y, dim_z],
            can be a numpy array or tensor
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
          box_corners_norm: corners as a percentage of the map size, in the
            format N x [x1, y1, x2, y2]. Origin is the top left corner
    """
    # tensor_format = isinstance(anchors, tf.Tensor)
    #
    # if not tensor_format:
    anchors = np.asarray(anchors)

    x = anchors[:, 0]
    z = anchors[:, 2]
    half_dim_x = anchors[:, 3] / 2.0
    half_dim_z = anchors[:, 5] / 2.0

    # Calculate extent ranges
    bev_x_extents_min = bev_extents[0][0]
    bev_z_extents_min = bev_extents[1][0]
    bev_x_extents_max = bev_extents[0][1]
    bev_z_extents_max = bev_extents[1][1]
    bev_x_extents_range = bev_x_extents_max - bev_x_extents_min
    bev_z_extents_range = bev_z_extents_max - bev_z_extents_min

    # 2D corners (top left, bottom right)
    x1 = x - half_dim_x
    x2 = x + half_dim_x
    # Flip z co-ordinates (origin changes from bottom left to top left)
    z1 = bev_z_extents_max - (z + half_dim_z)
    z2 = bev_z_extents_max - (z - half_dim_z)

    # Stack into (N x 4)
    # if tensor_format:
    #     bev_box_corners = tf.stack([x1, z1, x2, z2], axis=1)
    # else:
    bev_box_corners = np.stack([x1, z1, x2, z2], axis=1)

    # Convert from original xz into bev xz, origin moves to top left
    bev_extents_min_tiled = [bev_x_extents_min, bev_z_extents_min,
                             bev_x_extents_min, bev_z_extents_min]
    bev_box_corners = bev_box_corners - bev_extents_min_tiled

    # Calculate normalized box corners for ROI pooling
    extents_tiled = [bev_x_extents_range, bev_z_extents_range,
                     bev_x_extents_range, bev_z_extents_range]
    bev_box_corners_norm = bev_box_corners / extents_tiled

    return bev_box_corners, bev_box_corners_norm


def project_to_image_space(anchor, P, image_shape):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format 1 x [x, y, z,
            l, w, h, ry]
        stereo_calib_p2: stereo camera calibration p2 matrix
        image_shape: dimensions of the image [h, w]

    Returns:
        img_box: corners in image space - 1 x [x1, y1, x2, y2]
        box_norm: corners as a percentage of the image size -
            1 x [x1, y1, x2, y2]
    """
    # if anchors.shape[1] != 6:
    #     raise ValueError("Invalid shape for anchors {}, should be "
    #                      "(N, 6)".format(anchors.shape[1]))
    #

    pts_2d, pts_3d = compute_numpy_boxes_3d(anchor, P)

    if pts_2d is None:
        return None, None

    # Get the min and maxes of image coordinates
    x1 = np.amin(pts_2d[:, 0])
    y1 = np.amin(pts_2d[:, 1])

    x2 = np.amax(pts_2d[:, 0])
    y2 = np.amax(pts_2d[:, 1])

    img_box = np.array([x1, y1, x2, y2])

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    # Truncate remaining boxes into image space
    if img_box[0] < 0:
        img_box[0] = 0
    if img_box[1] < 0:
        img_box[1] = 0
    if img_box[2] > image_shape_w:
        img_box[2] = image_shape_w
    if img_box[3] > image_shape_h:
        img_box[3] = image_shape_h

    image_shape_tiled = [image_shape_w, image_shape_h,
                         image_shape_w, image_shape_h]

    box_norm = img_box / image_shape_tiled

    return np.array(img_box, dtype=np.float32), \
        np.array(box_norm, dtype=np.float32)




