''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import time
from PIL import Image
from _sys_init import root_dir
sys.path.append(os.path.join(root_dir(), 'lib', 'mayavi'))
import lib.dataset.kitti_util as utils
#from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.plane_dir = os.path.join(self.split_dir, 'planes')
        self.gt_depth_dir = os.path.join(self.root_dir, '..', 'data_scene_flow/training/disp_noc_0')
        self.pred_depth_dir = os.path.join(self.root_dir, '..', 'predict/pred')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples) 
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        # this is for KITTI/object/...
        assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert(idx<self.num_samples) 
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training') 
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)

    def get_ground_plane(self, idx):
        assert (idx < self.num_samples)
        plane = utils.get_road_plane(idx, self.plane_dir)
        return plane

    def get_gt_depth_map(self, idx):
        assert (idx < self.num_samples and self.split == 'training')
        gt_depth_filename = os.path.join(self.gt_depth_dir, str(idx).zfill(6) + "_10.png")
        print(gt_depth_filename)
        return utils.load_gt_depth_map(gt_depth_filename)

    def get_pred_depth_map(self, idx):
        assert(idx<self.num_samples and self.split=='training')
        img = self.get_image(idx)
        height, width, channels = img.shape

        pred_depth_filename = os.path.join(self.pred_depth_dir, '%06d_disp.npy'%(idx))
        return utils.load_pred_depth_map(pred_depth_filename, width, height)

    def get_top_down(self, idx):
        pass

class kitti_object_video(object):
    ''' Load data for KITTI videos '''
    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted([os.path.join(img_dir, filename) \
            for filename in os.listdir(img_dir)])
        self.lidar_filenames = sorted([os.path.join(lidar_dir, filename) \
            for filename in os.listdir(lidar_dir)])
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        #assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples) 
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        assert(idx<self.num_samples) 
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib

def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
    dataset = kitti_object_video(\
        os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
        os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
        video_path)
    print(len(dataset))
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        Image.fromarray(img).show()
        draw_lidar(pc)
        raw_input()
        pc[:,0:3] = dataset.get_calibration().project_velo_to_rect(pc[:,0:3])
        draw_lidar(pc)
        raw_input()
    return

def show_image_with_boxes(img, objects, calib, show3d=True, save_figure=False, save_figure_dir='',
                          img_name=''):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img) # for 2d bbox
    img2 = img.copy() # for 3d bbox
    for obj in objects:
        if isinstance(obj, np.ndarray):
            box3d_pts_2d, box3d_pts_3d = utils.compute_numpy_boxes_3d(obj, calib.P)
        else:
            if obj.type=='DontCare':continue
            cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
                (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            height, width, _ = img.shape
            # box3d_pts_2d[:, 0] = np.clip(box3d_pts_2d[:, 0], 0, height-1)
            # box3d_pts_2d[:, 1] = np.clip(box3d_pts_2d[:, 1], 0, width-1)
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
    if not isinstance(objects, np.ndarray):
        Image.fromarray(img1).show()
    if show3d:
        # img2_tmp = Image.fromarray(img2)
        # img2_tmp.show()
        cv2.imshow(img_name, img2)
        if save_figure:
            if save_figure_dir != '':
                save_figure_dir = os.path.join(root_dir(), save_figure_dir)
                print(save_figure_dir)
            if not os.path.exists(save_figure_dir):
                os.makedirs(save_figure_dir)
                print("done!!!!")
            filename = os.path.join(save_figure_dir, img_name)
            img2 = cv2.resize(img2, (1242,376), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(filename, img2)
        # img2_tmp.close()
        time.sleep(0.03)
        # cv2.destroyAllWindows()

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def old_get_lidar_in_area_extent(pc_velo, calib, area_extent):
    ''' Filter lidar points, keep those in area_extent '''
    pts = calib.project_velo_to_rect(pc_velo)
    if area_extent is not None:
        # Check provided extents
        extents_transpose = np.array(area_extent).transpose()
        if extents_transpose.shape != (2, 3):
            raise ValueError("Extents are the wrong shape {}".format(area_extent.shape))
        extent_inds = (pts[:,0] >= extents_transpose[0, 0]) & (pts[:,0]<extents_transpose[1, 0]) & \
                      (pts[:,1] >= extents_transpose[0, 1]) & (pts[:,1]<extents_transpose[1, 1]) & \
                      (pts[:,2] >= extents_transpose[0, 2]) & (pts[:,2]<extents_transpose[1, 2])

        return pts[extent_inds], extent_inds

def get_lidar_in_area_extent(pc_rect, area_extent):
    ''' Filter lidar points in camera's coordinate, keep those in area_extent '''
    if area_extent is not None:
        # Check provided extents
        extents_transpose = np.array(area_extent).transpose()
        if extents_transpose.shape != (2, 3):
            raise ValueError("Extents are the wrong shape {}".format(area_extent.shape))
        extent_inds = (pc_rect[:,0] >= extents_transpose[0, 0]) & (pc_rect[:,0]<extents_transpose[1, 0]) & \
                      (pc_rect[:,1] >= extents_transpose[0, 1]) & (pc_rect[:,1]<extents_transpose[1, 1]) & \
                      (pc_rect[:,2] >= extents_transpose[0, 2]) & (pc_rect[:,2]<extents_transpose[1, 2])

        return pc_rect[extent_inds], extent_inds

def get_lidar_in_img_fov_and_area_extent(pc_velo, calib, xmin, ymin, xmax, ymax,
                                        area_extent, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV  and area_extent'''
    pc_rect = calib.project_velo_to_rect(pc_velo)
    pts_2d = calib.project_rect_to_image(pc_rect)

    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    valid_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    # print("total points: %d"%(pc_velo.shape[0]))
    # print("points number in img_fov: %d" % len(valid_inds[valid_inds == 1]))
    if area_extent is not None:
        # Check provided extents
        extents_transpose = np.array(area_extent).transpose()
        if extents_transpose.shape != (2, 3):
            raise ValueError("Extents are the wrong shape {}".format(area_extent.shape))
        extent_inds = (pc_rect[:, 0] >= extents_transpose[0, 0]) & (pc_rect[:, 0] < extents_transpose[1, 0]) & \
                      (pc_rect[:, 1] >= extents_transpose[0, 1]) & (pc_rect[:, 1] < extents_transpose[1, 1]) & \
                      (pc_rect[:, 2] >= extents_transpose[0, 2]) & (pc_rect[:, 2] < extents_transpose[1, 2])
        valid_inds = valid_inds & extent_inds
        # print("points number in area_extents: %d"%(len(extent_inds[extent_inds==1])))
        # print("points left: %d"%len(valid_inds[valid_inds==1]))
    return pc_rect[valid_inds], valid_inds


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None): 
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    # from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
            tube_radius=None, line_width=1, figure=fig)
    mlab.show(1)

def show_lidar_with_numpy_boxes(pc_rect, objects, calib, save_figure, save_figure_dir='', img_name='',
                          img_fov=False, img_width=None, img_height=None, color=(1,1,1)):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d


    pc_velo = calib.project_rect_to_velo(pc_rect)

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0.5,0.5,0.5),
        fgcolor=None, engine=None, size=(1600, 1000))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)
    for obj in objects:
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_numpy_boxes_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_numpy_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, draw_text=False)
        mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.8, 0.8, 0.8),
                    tube_radius=None, line_width=1, figure=fig)
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance='auto', figure=fig)
    mlab.view(azimuth=180, elevation=60, focalpoint=[12.0909996, -1.04700089, 5.03249991], distance=62.0, figure=fig)
    if save_figure:
        if save_figure_dir != '':
            save_figure_dir = os.path.join(root_dir(), save_figure_dir)
            print(save_figure_dir)
        if not os.path.exists(save_figure_dir):
            os.makedirs(save_figure_dir)
            print("done!!!!")
        filename = os.path.join(save_figure_dir, img_name)
        mlab.savefig(filename)
    time.sleep(0.03)
    # mlab.close()


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show() 
    return img

def dataset_viz():
    dataset = kitti_object(os.path.join(ROOT_DIR, 'datasets/KITTI/object'))

    for data_idx in range(len(dataset)):
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(data_idx)[:,0:3]
        calib = dataset.get_calibration(data_idx)

        # Draw 2d and 3d boxes on image
        show_image_with_boxes(img, objects, calib, False)
        raw_input()
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        raw_input()

if __name__=='__main__':
    import mayavi.mlab as mlab
    # from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d
    dataset_viz()
