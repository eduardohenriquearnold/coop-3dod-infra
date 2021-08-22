import torch
import torch.utils.data
import numpy as np
import h5py

from lib.dataset.voxel_grid import VoxelGrid

class Transform():
    def __init__(self, x,y,z, yaw,roll,pitch):
        self.x,self.y,self.z, self.yaw,self.roll,self.pitch = x,y,z, yaw,roll,pitch
        self.setMatrix()

    def setMatrix(self):
        # Transformation matrix
        self.matrix = np.matrix(np.identity(4))
        cy = np.cos(self.yaw)
        sy = np.sin(self.yaw)
        cr = np.cos(self.roll)
        sr = np.sin(self.roll)
        cp = np.cos(self.pitch)
        sp = np.sin(self.pitch)
        self.matrix[0, 3] = self.x
        self.matrix[1, 3] = self.y
        self.matrix[2, 3] = self.z
        self.matrix[0, 0] =  (cp * cy)
        self.matrix[0, 1] =  (cy * sp * sr - sy * cr)
        self.matrix[0, 2] = - (cy * sp * cr + sy * sr)
        self.matrix[1, 0] =  (sy * cp)
        self.matrix[1, 1] =  (sy * sp * sr + cy * cr)
        self.matrix[1, 2] =  (cy * sr - sy * sp * cr)
        self.matrix[2, 0] =  (sp)
        self.matrix[2, 1] = - (cp * sr)
        self.matrix[2, 2] =  (cp * cr)
        return self.matrix

    def transform_points(self, points, inverse=False):
        """
        Given a 4x4 transformation matrix, transform an array of 3D points.
        Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
        """

        matrix = self.matrix if not inverse else np.linalg.inv(self.matrix)

        # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
        # the point matrix.
        points = points.transpose()
        # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        # Point transformation
        points = matrix * points
        # Return all but last row
        return points[0:3].transpose()


class Camera:
    def __init__(self, intrinsicMat, extrinsicMat):
        self.intrinsic = intrinsicMat
        self.extrinsic = extrinsicMat

        self.extTransform = Transform(0,0,0,0,0,0)
        self.extTransform.matrix = self.extrinsic

    def toCamera(self, points, inverse=False):
        '''Transforms points to camera reference (3D). Points formar: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]'''

        points = self.extTransform.transform_points(points, inverse)
        return points

    def toImagePlane(self, points):
        '''Given points in world reference, transform them to camera reference and then to image plane'''

        ptsCamera = self.toCamera(points).transpose()
        ptsImagePlane = np.dot(self.intrinsic, ptsCamera).transpose()

        #Remove points that are not in front of the camera
        idx = np.array(ptsImagePlane[:,2] > 0).squeeze().reshape(-1)
        ptsImagePlane = ptsImagePlane[idx]

        #Normalize to Z
        ptsImagePlane[:, 0] /= ptsImagePlane[:, 2]
        ptsImagePlane[:, 1] /= ptsImagePlane[:, 2]

        #Y axis has oposite direction on camera model
        return ptsImagePlane[:,0:2]

    def genPCL(self, depth):
        '''Given a depth frame generates a pointcloud with format [X,Y,Z] (shape n x 3) at camera ref. frame'''
        r = 2
        H, W = depth.shape 
        u = np.arange(0,W,r)
        v = np.arange(0,H,r)
        uu, vv = np.meshgrid(u,v)

        #Xc is given by the depth camera. Yc and Zc are calculated with the formulas
        #Yc = (u-cy)Xc/fy   Zc = (v-cz)Xc/fz
        cy, cz = self.intrinsic[0,0], self.intrinsic[1,0]
        f = self.intrinsic[0,1]

        #Introduce noise in the depth channel (sigma=0.05 according to HDL64E manual)
        #  noise = np.random.normal(0, scale=0.05, size=(H,W))
        #  depth += noise

        #Depth cam noise
        #  noise = np.random.normal(0, scale=0.01, size=(H,W))
        #  noise *= np.square(depth)
        #  depth += noise

        Xc = depth[::r,::r]
        Yc = (uu-cy)*Xc/f
        Zc = -(vv-cz)*Xc/f

        pts = np.hstack([i.reshape(-1,1) for i in (Xc,Yc,Zc)])
        return pts

class CooperativeDataset(torch.utils.data.Dataset):
    def __init__(self, path, ref, extents, voxSize, maxPtsVox, augment=False, selectCameras=None, vext=None):
        '''Creates a Dataset object. The object locations and 3d point clouds are relative to the *ref* given. Objects outside the x,y,z extent ranges ((xmin,xmax),(ymin,ymax),(zmin,zmax)) are deleted'''

        #Opens dataset H5 file
        self.dpath = path
        data = h5py.File(path, 'r')

        self.length = data['dslices'].shape[0]
        self.augment = augment

        #Get cameras 
        cameras = list(data.get('cameras'))

        #Set reference frame (X,Y,Z) and extent
        self.ref = torch.FloatTensor(ref)
        self.ext = torch.FloatTensor(extents) 
        self.vext = self.ext.clone() if vext is None else torch.FloatTensor(vext)

        #Set voxelsize and maxPts in vox
        self.voxSize = voxSize
        self.maxPtsVox = maxPtsVox

        #Get cameras 
        self.cameras = []
        for cam in cameras:
            intrinsic = np.array(data.get(f'cameras/{cam}/intrinsic'))
            extrinsic = np.array(data.get(f'cameras/{cam}/extrinsic'))
            if cam[0] == 'd':
                self.cameras.append(Camera(intrinsic, extrinsic))
        data.close()

        #Select cameras
        self.selectedCameras = selectCameras

        #Radius-based filtering (hybrid fusion)
        self.R = None

    def __len__(self):
        '''Size of the dataset is the number of slices'''

        return self.length

    def getPCL(self, imgs):
        '''Generate fused PCL given depth images'''

        pts = []
        for i, cam in enumerate(self.cameras):
            #Ignore camera if not in selected cameras
            if self.selectedCameras is not None:
                if i not in self.selectedCameras:
                    continue

            #Get pcl in the camera reference frame
            p = cam.genPCL(imgs[i])

            #Keep points outside radius threshold R
            if self.R:
                r = p[:,0]**2+p[:,1]**2
                idx = r > self.R**2
                p = p[idx]

            #Transform pts to global reference frame and fix for reference
            p = np.matrix(p)
            p = cam.toCamera(p, inverse=True)
            #Add pts to array
            pts.append(p)

        #Concatenate points from all frames
        pts = np.concatenate(pts, axis=0)
        pts = torch.FloatTensor(pts)

        #Transform to KITTI reference frame (Z=Y, Y=Z)
        pts = pts[:,[0,2,1]]

        #Transform points to the reference frame
        pts -= self.ref.view(1,3)

        return pts

    def getBoxes(self, boxes):
        '''Generate boxes in the right format'''

        #Filter only vehicles (cls id 0)
        cls = boxes[:,-1]
        boxes = boxes[cls==0]

        #Transform boxes to KITTI ref
        boxes = boxes[:,[0,2,1,3,4,5,6,7,8,9]]

         #Get boxes in the reference frame
        boxes[:,:3] -= self.ref.view(1,3)

        #Get the representation to the network format (x,y,z,w,l,h,yaw)
        xyz=boxes[:,:3]
        wlh=2*boxes[:,6:9]
        yaw=boxes[:,5].view(-1,1)*np.pi/180
        boxes = torch.cat([xyz,wlh,yaw], dim=1)

        return boxes

    def getPts(self, pcl, gtbox):
        '''Given a point-cloud and gt-box, get the idx of points within the gt box'''

        x,y,z = gtbox[0:3] 
        yaw = gtbox[6]
        ex, ez, ey = 0.5*gtbox[3:6] 
        trActor = Transform(x,y,z,0,0,yaw)
        ptsInActorCS = torch.FloatTensor(trActor.transform_points(np.matrix(pcl[:,:3]), inverse=True))

        #get pts inside box (offset on the y-axis because some bb limits are very close)
        idx = (np.absolute(ptsInActorCS[:,0])<ex)*(np.absolute(ptsInActorCS[:,1])<(ey+0.1))*(np.absolute(ptsInActorCS[:,2])<ez)
        return idx, ptsInActorCS[idx] 

    def augmentate_sample_pts(self, pts, boxes, T):
        '''Given points, keep only T random pts from each vehicle.'''

        pts_new = pts[:]
        
        #Remove all pts keeping record of num of pts per vehicle
        for box in boxes:
            idx, ptsVeh = self.getPts(pts_new, box)

            #Shuffle points
            ptsOCS = np.copy(pts_new[idx])
            np.random.shuffle(ptsOCS)

            #Sample T random pts
            ptsOCS = ptsOCS[:T]

            #Remove pts
            pts_new = pts_new[1-idx]

            if ptsOCS.shape[0] == 0:
                continue

            #Add to the pt cloud 
            pts_new = np.concatenate((pts_new, ptsOCS))
            pts_new = torch.FloatTensor(pts_new)

       #Return new pts and single box
        return pts_new, boxes 

    def augmentate(self, pts, boxes):
        '''Apply angle and translation augmentation to all boxes'''

        nboxes = []

        #Transform each vehicle
        for box in boxes:
            #Get index of points and pts in vehicle coordinate system
            idxpts, ptsVeh = self.getPts(pts, box)
            if ptsVeh.shape[0] == 0:
                continue

            #Add random angle and random displacement
            nbox = box
            nbox[0] += np.random.normal(scale=0.5)
            nbox[2] += np.random.normal(scale=0.5)
            nbox[6] += np.random.uniform(-np.pi/10,np.pi/10)
            nboxes.append(nbox.reshape(1,-1))

            #Generate transform, apply it and replace points in the pointcloud
            tr = Transform(nbox[0], nbox[1], nbox[2], 0, 0, nbox[6])
            ptsVehGlob = torch.FloatTensor(tr.transform_points(np.matrix(ptsVeh), inverse=False))
            pts[idxpts] = ptsVehGlob

        #Concatenate new gt
        nboxes = np.concatenate(nboxes)
        boxes = torch.FloatTensor(nboxes)

        return pts, boxes

    def augmentate_ang(self, pts, boxes):
        '''Apply rotation to whole PCL around central point'''

        #Ref is at central point
        ref = self.ext[:,1]-self.ext[:,0]
        ref[1] = 0
        ref /= 2

        #Transform all points to ref CS 
        tt = Transform(ref[0], ref[1], ref[2], 0,0,0)
        pts = tt.transform_points(np.matrix(pts), inverse=True)

        #Rotate them by random angle and get back to original coordinate system
        ang = np.random.uniform(-np.pi/4, np.pi/4)
        tr = Transform(0,0,0,0,0,ang)
        pts = tr.transform_points(pts) 
        pts = tt.transform_points(pts)
        pts = torch.FloatTensor(pts)

        #Get new boxes position and angles
        bpos = boxes[:,:3]
        bpos = tt.transform_points(np.matrix(bpos), inverse=True)
        bpos = tr.transform_points(bpos)
        bpos = tt.transform_points(bpos)
        boxes[:,:3] = torch.FloatTensor(bpos)
        boxes[:,-1] += ang

        return pts, boxes

    def __getitem__(self, key):
        '''Returns items corresponding to the slice index key.''' 

        #Load file and get depth images and ground-truth information
        with h5py.File(self.dpath, 'r') as data:
            #Get depth images in torch format and create new dim (for diff cameras)
            imgs = np.array(data['dslices'][key])

            #Get ground-truth object representations
            boxes = torch.FloatTensor(data.get(f'objects/{key}'))

        #Generate pcl
        pts = self.getPCL(imgs)

        #Get boxes
        boxes = self.getBoxes(boxes)

        #Augment 
        if self.augment:
            pts, boxes = self.augmentate(pts, boxes)
            #  pts, boxes = self.augmentate_ang(pts, boxes)

        #Certify pts and boxes are within the extent of detector
        idx = (pts[:,0]>self.ext[0,0])*(pts[:,0]<self.ext[0,1])*(pts[:,1]>self.ext[1,0])*(pts[:,1]<self.ext[1,1])*(pts[:,2]>self.ext[2,0])*(pts[:,2]<self.ext[2,1])
        pts = pts[idx]
        idx = (boxes[:,0]>self.ext[0,0])*(boxes[:,0]<self.ext[0,1])*(boxes[:,2]>self.ext[2,0])*(boxes[:,2]<self.ext[2,1])
        boxes = boxes[idx]

        #If boxes is empty, fill it all with zeros (prevent error in Dataloader)
        if boxes.shape[0] == 0:
            boxes = torch.zeros(1,7)

        #Generate voxelized version of pcl
        voxGrid = VoxelGrid()
        voxGrid.voxelize(pts.numpy(), self.voxSize, extents=self.vext, num_T=self.maxPtsVox)
       
        return [boxes, voxGrid.points, voxGrid.unique_indices, voxGrid.num_pts_in_voxel, voxGrid.leaf_layout, voxGrid.voxel_indices, voxGrid.padded_voxel_points, voxGrid.num_divisions]

class DataLoader(torch.utils.data.DataLoader):
    '''Create an alternative loader that merges imgs and vs as tensors but keep boxes as list of tensors, since it cannot placed into single tensor due to differnet number of boxes per slice'''
    def __init__(self, *arg1,**arg2):
        super().__init__(*arg1, **arg2, collate_fn = self._collate_fn)
    
    def _collate_fn(self, batch):

        batch_size = len(batch)
        zip_batch = list(zip(*batch))
        ground_truth_bboxes_3d = zip_batch[0]
        s_points = zip_batch[1]
        unique_indices = zip_batch[2]
        num_pts_in_voxel = zip_batch[3]
        leaf_out = zip_batch[4]
        s_voxel_indices = zip_batch[5]
        s_voxel_points = zip_batch[6]
        num_divisions = zip_batch[7]
        img_ids = []

        max_num_gt_bboxes_3d = max([_.shape[0] for _ in ground_truth_bboxes_3d])
        max_points = max([_.shape[0] for _ in s_points])
        max_indices = max([_.shape[0] for _ in unique_indices])
        # max_num_pts = max([_.shape[0] for _ in num_pts_in_voxel])
        # max_voxel_indices = max([_.shape[0] for _ in s_voxel_indices])

        padded_images = []
        padded_gt_bboxes_2d = []
        padded_gt_bboxes_3d = []
        padded_points = []
        padded_indices = []
        padded_num_pts = []
        padded_voxel_indices = []
        padded_voxel_points = []

        for b_ix in range(batch_size):
            # pad zeros to gt_bboxes
            gt_bboxes_3d = ground_truth_bboxes_3d[b_ix]
            new_gt_bboxes_3d = np.zeros([max_num_gt_bboxes_3d, gt_bboxes_3d.shape[-1]])
            new_gt_bboxes_3d[:gt_bboxes_3d.shape[0], :] = gt_bboxes_3d
            padded_gt_bboxes_3d.append(new_gt_bboxes_3d)

            points = s_points[b_ix]
            new_points = np.zeros([max_points, points.shape[-1]])
            new_points[:points.shape[0], :] = points
            padded_points.append(new_points)

            indices = unique_indices[b_ix]
            new_indices = np.zeros([max_indices])
            new_indices[:indices.shape[0]] = indices
            padded_indices.append(new_indices)

            num_pts = num_pts_in_voxel[b_ix]
            new_num_pts = np.zeros(max_indices)
            new_num_pts[:num_pts.shape[0]] = num_pts
            padded_num_pts.append(new_num_pts)

            voxel_indices = s_voxel_indices[b_ix]
            new_voxel_indices = np.zeros([max_indices, voxel_indices.shape[-1]+1], dtype=np.int64)
            new_voxel_indices[:voxel_indices.shape[0], 1:] = voxel_indices
            new_voxel_indices[:voxel_indices.shape[0], 0] = b_ix
            padded_voxel_indices.append(new_voxel_indices)

            voxel_points = s_voxel_points[b_ix]
            new_voxel_points = np.zeros([max_indices, voxel_points.shape[-2], voxel_points.shape[-1]], dtype=np.float32)
            new_voxel_points[:voxel_points.shape[0], :] = voxel_points
            padded_voxel_points.append(new_voxel_points)

        padded_images = torch.FloatTensor()
        padded_gt_bboxes_2d = torch.FloatTensor() 
        ground_plane = torch.FloatTensor()
        if max_num_gt_bboxes_3d == 0:
            padded_gt_bboxes_3d = torch.from_numpy(np.zeros([0, 7]))
        else:
            padded_gt_bboxes_3d = torch.from_numpy(np.stack(padded_gt_bboxes_3d, axis = 0))
        padded_points = torch.from_numpy(np.stack(padded_points, axis= 0))
        padded_indices = torch.from_numpy(np.stack(padded_indices, axis= 0))
        padded_num_pts = torch.from_numpy(np.stack(padded_num_pts, axis= 0))
        leaf_out = torch.from_numpy(np.array(leaf_out))
        padded_voxel_indices = torch.from_numpy(np.stack(padded_voxel_indices, axis=0))
        # voxel = torch.from_numpy(np.array(s_voxel))
        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points, axis=0))
        num_divisions = np.asarray(num_divisions)


        return padded_images, padded_points, padded_indices, padded_num_pts, \
               leaf_out, padded_voxel_indices, padded_voxel_points, ground_plane, \
               padded_gt_bboxes_2d, padded_gt_bboxes_3d, img_ids, num_divisions

def getBBvertices(boxes):
    '''Given box encoding shape [N,7] (x,y,z,w,l,h,yaw) generates BB vertices shaped [N,8,3]'''

    v = torch.zeros(boxes.shape[0], 8, 3)

    for i,b in enumerate(boxes):
        x,y,z = b[0:3]
        w,h,l = 0.5*b[3:6]
        yaw = b[6]
        tr = Transform(x,y,z,0,0,yaw)

        bbox_pts = np.array([
        [  w,   l,   h],
        [- w,   l,   h],
        [- w, - l,   h],
        [  w, - l,   h],
        [  w,   l, - h],
        [- w,   l, - h],
        [- w, - l, - h],
        [  w, - l, - h]
        ])

        bbox_pts = tr.transform_points(bbox_pts)
        v[i] = torch.FloatTensor(np.array(bbox_pts))

    return v

