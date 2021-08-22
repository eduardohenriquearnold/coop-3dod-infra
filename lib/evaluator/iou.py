import torch

def rotMat(ang):
    '''Get rotation matrix given ang (rad)'''

    return torch.FloatTensor(((torch.cos(ang), -torch.sin(ang)),(torch.sin(ang),torch.cos(ang))))

def iou(b1, b2, n=50, bv=True):
    '''Given boxes arrays b1 and b2, approximate IOU using samling with n x n grid of points.
    Input format: [box_idx, 8], where the 8 dims are the bb encoding (check RPN)
    Does not take y-dimension into account (BEV)'''
    
    assert b1.size(0)==b2.size(0), "Number of boxes dont match"
    assert b1.size(1)==b2.size(1)==7, "Number of boxes parameters dont match"
    assert b1.dim()==b2.dim()==2, "Number of dimensions of boxes differ from the expected (2)"

    #B1 will be static and have sides parallel to axis. B2 will be rotated by the relative angle between them
    #This facilitates checking if points sampled from B2 are within B1.
    ptsB2 = torch.zeros((b1.size(0), 2, n*n)).to(b1.device)
    aB1 = b1[:,6]
    aB2 = b2[:,6] 

    #Check if boxes intersect
    centreDst = torch.sqrt((b1[:,0]-b2[:,0]).pow(2)+(b1[:,2]-b2[:,2]).pow(2))
    d1 = torch.sqrt((b1[:,3]/2).pow(2)+(b1[:,4]/2).pow(2))
    d2 = torch.sqrt((b2[:,3]/2).pow(2)+(b2[:,4]/2).pow(2))
    nIntersect = centreDst >= d1+d2

    for i in range(b1.size(0)):
        #Skip if boxes that do not intersect
        if nIntersect[i]:
            continue

        #Create grid of points
        w,l = b2[i,3], b2[i,4]
        ptsB2[i,0,:] = torch.linspace(-w/2, w/2, n, device=b1.device).repeat(n)  #Repeats array after array 123,123,123
        ptsB2[i,1,:] = torch.linspace(-l/2, l/2, n, device=b1.device).repeat(n).reshape(n,-1).transpose(1,0).reshape(-1) #Repeats column after column 111,222,333

        #Rotate them as B2
        rotM = rotMat(aB2[i]).to(b1.device)
        ptsB2[i] = torch.matmul(rotM, ptsB2[i])

        #Transpose them relative to B1
        ptsB2[i,0] += b2[i,0] - b1[i,0]
        ptsB2[i,1] += b2[i,2] - b1[i,2]

        #Rotate again, now correcting for B1 angle
        rotM = rotMat(-aB1[i]).to(b1.device)
        ptsB2[i] = torch.matmul(rotM, ptsB2[i])

    #Check how many points of B2 fall within B1
    wb1 = b1[:,3].repeat(n*n).reshape(n*n,-1).transpose(1,0).reshape(-1, n*n) #torch.repeat in order
    lb1 = b1[:,4].repeat(n*n).reshape(n*n,-1).transpose(1,0).reshape(-1, n*n)
    cX = torch.abs(ptsB2[:,0]) <= wb1/2
    cY = torch.abs(ptsB2[:,1]) <= lb1/2
    iA = (cX*cY).float().mean(dim=1) #relative intersection area (relative to B2 area)
    iA *= b2[:,3]*b2[:,4]            #absolute intersection area

    #Calculates IOU=(intersection)/(area1+area2-intersection)
    u = b1[:,3]*b1[:,4] + b2[:,3]*b2[:,4] - iA
    iou = iA/u

    #Considers the 3D case
    if not bv:
        b1hmax = b1[:,1] + 0.5*b1[:,5]
        b1hmin = b1[:,1] - 0.5*b1[:,5]
        b2hmax = b2[:,1] + 0.5*b2[:,5]
        b2hmin = b2[:,1] - 0.5*b2[:,5]

        htp = torch.min(b1hmax, b2hmax)
        hbt = torch.max(b1hmin, b2hmin) 
        intLength = htp-hbt
        maxLength = torch.max(b1[:,5], b2[:,5])

        hInt = torch.max(torch.zeros_like(intLength), intLength)/maxLength
        iou *= hInt

    #Discards points where there is no intersection
    iou[nIntersect] = 0
    return iou
