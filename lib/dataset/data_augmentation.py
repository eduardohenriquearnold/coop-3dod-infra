# data augmentation for kitti, according to voxelnet.

import numpy as np
from lib.dataset.kitti_util import roty

def augmentation_boxes():
    pass

def augmentation_scale_all(points, boxes_3d):
    # scale on all points and boxes
    assert points.shape[-1] == 4
    assert boxes_3d.shape[-1] == 7
    seed = np.random.uniform(0.95, 1.05, 1)
    # seed = np.random.uniform(1.3, 2.05, 1)
    aug_points = np.zeros_like(points)
    aug_boxes_3d = np.zeros_like(boxes_3d)

    aug_points[:, :3] = points[:, :3] * seed
    aug_points[:, -1] = points[:, -1]
    aug_boxes_3d[:, :6] = boxes_3d[:, :6] * seed
    aug_boxes_3d[:, -1] = boxes_3d[:, -1]

    return aug_points, aug_boxes_3d, seed

def augmentation_rotation_all(points, boxes_3d):
    # rotation on all points and boxes
    # (R*P.T).T = P * R.T
    assert points.shape[-1] == 4
    assert boxes_3d.shape[-1] == 7
    seed = np.random.uniform(-np.pi/4, np.pi/4, 1)
    R = roty(seed)
    aug_points = np.zeros_like(points)
    aug_boxes_3d = np.zeros_like(boxes_3d)
    aug_points[:, :3] = np.dot(points[:, :3], R.T)
    aug_points[:, -1] = points[:, -1]
    aug_boxes_3d[:, :3] = np.dot(boxes_3d[:, :3], R.T)
    aug_boxes_3d[:, 3:6] = boxes_3d[:, 3:6]
    aug_boxes_3d[:, -1] = boxes_3d[:, -1] + seed
    return aug_points, aug_boxes_3d, seed

def get_boxes_in_area_extent(bboxes_3d, area_extent):
    assert bboxes_3d.shape[-1] == 7
    if area_extent is not None:
        # Check provided extents
        extents_transpose = np.array(area_extent).transpose()
        if extents_transpose.shape != (2, 3):
            raise ValueError("Extents are the wrong shape {}".format(area_extent.shape))
        extent_inds = (bboxes_3d[:,0] >= extents_transpose[0, 0]) & (bboxes_3d[:,0]<extents_transpose[1, 0]) & \
                      (bboxes_3d[:,2] >= extents_transpose[0, 2]) & (bboxes_3d[:,2]<extents_transpose[1, 2])
        return bboxes_3d[extent_inds]
