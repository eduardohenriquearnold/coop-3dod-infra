import sys
import h5py
import numpy as np
import mayavi.mlab as mlab
import cv2
import argparse
from scipy.interpolate import RectBivariateSpline

parser = argparse.ArgumentParser(description="Visualise recorded data frames")
parser.add_argument("-p", dest="pointcloud", action="store_true", help="Visualise point-cloud" )
parser.add_argument("-l", dest="lidar", action="store_true", help="Visualise point-cloud as lidar" )
parser.add_argument("file", help="Storage file")
args = parser.parse_args()

class Transform():
    def __init__(self, x,y,z, yaw,roll,pitch):
        self.x,self.y,self.z, self.yaw,self.roll,self.pitch = x,y,z, yaw,roll,pitch
        self.setMatrix()

    def setMatrix(self):
        # Transformation matrix
        self.matrix = np.matrix(np.identity(4))
        cy = np.cos(np.radians(self.yaw))
        sy = np.sin(np.radians(self.yaw))
        cr = np.cos(np.radians(self.roll))
        sr = np.sin(np.radians(self.roll))
        cp = np.cos(np.radians(self.pitch))
        sp = np.sin(np.radians(self.pitch))
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
        '''Transforms points to camera reference (3D). Points format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]]'''

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

        return ptsImagePlane[:,0:2]

    def genPCL(self, depth):
        '''Given a depth frame generates a pointcloud with format [X,Y,Z] (shape n x 3) at camera ref. frame'''

        H, W = depth.shape 
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u,v)

        #Xc is given by the depth camera. Yc and Zc are calculated with the formulas
        #Yc = (u-cy)Xc/fy   Zc = (v-cz)Xc/fz
        cy, cz = self.intrinsic[0,0], self.intrinsic[1,0]
        f = self.intrinsic[0,1]

        Xc = depth
        Yc = (uu-cy)*Xc/f
        Zc = -(vv-cz)*Xc/f

        pts = np.hstack([i.reshape(-1,1) for i in (Xc,Yc,Zc)])
        return pts

    def genPCL_lidar(self, depth):
        '''Given a depth frame generates a pointcloud with format [X,Y,Z] (shape n x 3) at camera ref. frame assuming HDL64 lidar sampling points'''

        theta = np.deg2rad(np.arange(-45,45, 0.09))
        phi = np.deg2rad(np.arange(88, 114.9, 0.42))
        tt, pp = np.meshgrid(theta, phi)

        #create unit vectors from polar angles
        x = np.cos(tt)*np.sin(pp)
        y = np.sin(tt)*np.sin(pp)
        z = np.cos(pp)
        pts = np.hstack([i.reshape(-1,1) for i in (x,y,z)])
 
        #obtain projections of unit vector on image plane
        ptsIP= np.dot(self.intrinsic, pts.T).transpose()
        ptsIP[:, 0:2] /= ptsIP[:, 2].reshape(-1,1)
        ptsIP = ptsIP[:,:2]

        #interpolate depth values in the corresponding points
        depth_int = RectBivariateSpline(np.arange(depth.shape[0]), np.arange(depth.shape[1]), depth)
        uu = ptsIP[:,0]
        vv = ptsIP[:,1]
        depth = depth_int(vv,uu, grid=False)

        #Get final 3D pts
        x = depth
        y = depth*(np.tan(tt)).reshape(-1)
        z = depth*(np.cos(pp)/np.cos(tt)/np.sin(pp)).reshape(-1)
        pts = np.hstack([i.reshape(-1,1) for i in (x,y,z)])

        #  mlab.points3d(pts[:,0], pts[:,1], pts[:,2], mode='point')
        #  input()
        return pts
 
def getBBpts(repr):
    '''Given a representation get 8 BB edge points in world reference. Out shape [8,3]'''

    x,y,z = repr[0],repr[1],repr[2]
    pitch,roll,yaw = repr[3],repr[4],repr[5]
    trActor = Transform(x,y,z,yaw,pitch,roll)

    #BB Extensions (half length for each dimension)
    ex, ey, ez = repr[6],repr[7],repr[8]

    #Get 8 coordinates
    bbox_pts = np.array([
    [  ex,   ey,   ez],
    [- ex,   ey,   ez],
    [- ex, - ey,   ez],
    [  ex, - ey,   ez],
    [  ex,   ey, - ez],
    [- ex,   ey, - ez],
    [- ex, - ey, - ez],
    [  ex, - ey, - ez]
    ])

    #Transform the bbox points from actor ref frame to global reference
    bbox_pts = trActor.transform_points(bbox_pts)
    return bbox_pts

def drawPoints(im, pts):
    '''Draw points in an image'''
    for pt in pts:
        center = (int(pt[0,0]), int(pt[0,1]))
        cv2.circle(im, center, 3, 255, -1)

def drawActors(reprs, fig, line_width=1):
    '''Given array of n actor repr, generate vertices and plot lines in 3D vis'''

    colors = [(1,0,0),(0,1,0),(0,0,1)] #Vehicle is red. Cycle is Green. Pedestrian is Blue
    for r in reprs:
        b = getBBpts(r)
        cls = int(r[-1])
        color = colors[cls]
        for k in range(0,4):
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

#Load file
fname = args.file 
f = h5py.File(fname, 'r')

#Load Cameras
cameras_ = list(f.get('cameras'))
cameras  = []
dcameras = []
for cam in cameras_:
    intrinsic = np.array(f.get(f'cameras/{cam}/intrinsic'))
    extrinsic = np.array(f.get(f'cameras/{cam}/extrinsic'))
    if cam[0] == 'd':
        dcameras.append(Camera(intrinsic, extrinsic))
    else:
        cameras.append(Camera(intrinsic, extrinsic))


#Show slices
slices = f['slices']
dslices = f['dslices']
sliceNum = 0
while (sliceNum < slices.shape[0]):

    print(f"Slice {sliceNum}")
    
    #Point-cloud visualisation
    if args.pointcloud:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))

        #  np.random.seed(212)
        np.random.seed(216)
        for camNum, cam in enumerate(dcameras):
            #Get corresponding frame
            frame = np.array(dslices[sliceNum, camNum])

            #Get pseudo-point cloud in camera ref frame
            if args.lidar:
                pts = cam.genPCL_lidar(frame)
            else:
                pts = cam.genPCL(frame)

            #Transform pts to absolute coordinate system
            pts = np.matrix(pts)
            pts = cam.toCamera(pts, inverse=True)

            #Remove points with Z-coord above 3m and below 0m
            pts = np.array(pts)
            idx = (pts[:,2]>0)*(pts[:,2]<3)
            pts = pts[idx]

            #Seed for random colours
            color = tuple(np.random.rand(3).tolist())

            #Plot points
            mlab.points3d(pts[:,0], pts[:,1], pts[:,2], mode='point', color=color, figure=fig)

        #Get actors bbs
        actorsRepr = np.array(f.get(f'objects/{sliceNum}'))
        drawActors(actorsRepr, fig)

        try:
            mlab.show()
        except KeyboardInterrupt:
            sys.exit(0)

    #Normal frame visualisation
    else:
        #Get actors
        actorsRepr = np.array(f.get(f'objects/{sliceNum}'))
        bbpts = [getBBpts(r) for r in actorsRepr]

        for camNum, cam in enumerate(cameras):
            #Get corresponding frame
            frame = np.array(slices[sliceNum, camNum])

            #Draw all object points on the frame
            for objBBpts in bbpts:
                ptsImagePlane = cam.toImagePlane(objBBpts)
                drawPoints(frame, ptsImagePlane)

            #Show frame
            cv2.imshow(str(camNum), frame)

        for camNum, cam in enumerate(dcameras):
            #Get corresponding frame
            frame = np.array(dslices[sliceNum, camNum])
            frame /= 1000
            frame = np.log(80*frame+1)/np.log(81)

            #Show frame
            cv2.imshow('d'+str(camNum), frame)

        try:
            cv2.waitKey(0)
        except KeyboardInterrupt:
            sys.exit(0)

    inp = input()
    if inp=='':
        sliceNum += 1
    else:
        sliceNum = int(inp)

