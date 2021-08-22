#encoding: utf-8
import numpy as np
import logging
logger = logging.getLogger('global')

def get_anchors_over_plane(featmap_h, featmap_w, area_extents, anchor_3d_sizes, anchor_stride, ground_plane):

    # Convert sizes to ndarray
    anchor_3d_sizes = np.asarray(anchor_3d_sizes)

    anchor_stride_x = anchor_stride[0]
    anchor_stride_z = anchor_stride[1]
    anchor_rotations = np.asarray([0, np.pi / 2.0])

    x_start = area_extents[0][0] + anchor_stride[0] / 2.0
    x_end = area_extents[0][1]
    x_centers = np.array(np.arange(x_start, x_end, step=anchor_stride_x),
                         dtype=np.float32)

    # z_start = area_extents[2][1] - anchor_stride[1] / 2.0
    # z_end = area_extents[2][0]
    # z_centers = np.array(np.arange(z_start, z_end, step=-anchor_stride_z),
    #                      dtype=np.float32)
    z_start = area_extents[2][0] + anchor_stride[1] / 2.0
    z_end = area_extents[2][1]
    z_centers = np.array(np.arange(z_start, z_end, step=anchor_stride_z),
                         dtype=np.float32)
    logger.debug('x shape: %d, z shape: %d'%(len(x_centers), len(z_centers)))
    # get anchors on one grid
    anchors = get_anchors(anchor_3d_sizes, anchor_rotations)
    # spread anchors over each grid
    # shift_x = np.arange(0, featmap_w) * anchor_stride
    # shift_y = np.arange(0, featmap_h) * anchor_stride
    # [featmap_h, featmap_w]
    # shift_x, shift_z = np.meshgrid(x_centers, z_centers)
    shift_z, shift_x = np.meshgrid(z_centers, x_centers)
    shifts = np.vstack((shift_x.ravel(), shift_z.ravel())).transpose()

    # a, b, c, d = ground_plane
    all_x = shifts[:, 0]
    all_z = shifts[:, 1]
    # all_y = -(a * all_x + c * all_z + d) / b
    all_y = np.zeros([shifts.shape[0]])

    _shifts = np.vstack((all_x, all_y, all_z, np.zeros((4, all_y.shape[0])))).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    assert K == featmap_h * featmap_w
    anchors_overplane = (anchors.reshape((1, A, 7)) +
                    _shifts.reshape((1, K, 7)).transpose((1, 0, 2)))
    anchors_overplane = anchors_overplane.reshape((K * A, 7))
    anchors_overplane[:, 1] = anchors_overplane[:, 1] + anchors_overplane[:, -2]
    return anchors_overplane

def get_anchors(anchor_3d_sizes, anchor_rotations):
    """
    
    :param anchor_3d_sizes: n*[l,w,h]
    :param anchor_rotations: [0,pi/2,...]
    :return: 
    """
    #
    num_size, _ = anchor_3d_sizes.shape
    num_rotations = len(anchor_rotations)
    anchors = np.zeros([num_size, 7])
    #size
    for size, a in zip(anchor_3d_sizes, anchors):
        a[3:6] = size
    #rotation
    anchors = np.tile(anchors, [num_rotations, 1])
    for i in range(num_rotations):
        anchors[num_size*i:num_size*(i+1), 6] = anchor_rotations[i]
    # print("anchors :", anchors)
    return anchors



def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors

if __name__ == '__main__':
    anchor_3d_sizes = np.array([2,3,4,5,6,7]).reshape(-1,3)
    anchor_stride   = np.array([2])
    anchor_rotations = np.asarray([0, np.pi / 2.0])
    area_extents = np.array([-40, 40, -5, 3, 0, 70]).reshape(-1,2)
    get_anchors(anchor_3d_sizes, anchor_rotations)
    anchors = get_anchors_over_plane(0, 0, area_extents, anchor_3d_sizes, [2,2], [0,-1,0,1.6])
    print('done!')