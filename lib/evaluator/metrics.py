import torch
import numpy as np
from lib.evaluator.iou import iou

class MetricsCalculator:
    '''Easy class interface to calculate metrics'''

    def __init__(self, iou_threshold, bv=True):
        #Default AP is the one that uses all points for interpolation (Pascal VOC 2010+)
        self.AP = self.AP_allint

        #IoU thr
        self.iou_threshold = iou_threshold

        #BV
        self.bv = bv

        #Init values
        self.wipe()

    def wipe(self):
        '''Remove all previous detections'''

        self.scores = []
        self.tps = []
        self.nobjs = 0

    def getTPVector(self, pred, gt):
        '''Given a slice prediction and corresponding gt, returns a bool tensor indicating whether the prediction is a TP.
        To be a TP it has to have IoU higher than threshold. If multiple predictions are made for a single gt, only the one with highest probability is a TP.
        Assumes predictions are already sorted by probability in descending order.'''

        tp = torch.zeros(pred.size(0)).byte().to(pred.device)

        #Ignore score
        pred = pred[:,:7]

        #Calculates IoU of all pred and all gts
        predR = pred.repeat(1, gt.size(0)).reshape(-1, 7)
        gtR = gt.repeat(pred.size(0), 1)
        ious = iou(predR, gtR, bv=self.bv).reshape(pred.size(0), gt.size(0))      #Results in vector [n_pred, n_gt] of IOUs

        #Get max IOU for each pred
        mIOU, gIdx = torch.max(ious, dim=1)

        #Check IoU higher than threshold
        c1 = mIOU > self.iou_threshold

        #gIdx should also be unique, otherwise, only the first pred is valid (highest probability, assuming pred is sorted)
        #torch.unique does not have return_index argument which is required in this case
        _, idxPredsWithUniqueGT = np.unique(gIdx.cpu(), return_index=True)
        idxPredsWithUniqueGT = torch.from_numpy(idxPredsWithUniqueGT).to(pred.device)

        c2 = tp.clone()
        c2[idxPredsWithUniqueGT] = 1

        #TP only when both conditions are satisfied
        tp = c1*c2
        return tp

    def accumulate(self, pred, gt):
        '''Given a single frame prediction, accumulates the scores and tps vector for further PR calculation.'''

        pred = torch.FloatTensor(pred)
        gt = torch.FloatTensor(gt)

        assert pred.dim() == 2, 'Pred should contain predictions for a frame 2 dims: (nboxes, box_attr)'

        self.scores.append(pred[:,-1])
        self.tps.append(self.getTPVector(pred, gt))
        self.nobjs += len(gt)

    def PR(self):
        '''Calculates precision and recall vectors for batch of predictions, given gt and IOU threshold'''

        #Merges all accumulated detections
        scores = torch.cat(self.scores)
        tps = torch.cat(self.tps)

        #Reorders based on probabilities
        _, sidx = torch.sort(scores, descending=True)
        tps = tps[sidx]

        #precision at step i is number of TP up to step i divided by all objs detected up top step i
        #recall at step i is number of TP up to step i divided by the overall number of objects 
        cumtp = torch.cumsum(tps.float(), dim=0)
        pr = cumtp / torch.arange(1, cumtp.size(0)+1).float().to(scores.device)
        re = cumtp / self.nobjs

        return pr, re

    def AP_allint(self, pr,re):
        '''Calculates Average Precision (AP) given precision and recall vectors. Uses VOC PASCAL all points interpolation (after 2010) (ref https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision).
        AP = sum[(r_(n+1)-r_n) p'(r_(n+1))] for all r
        p'(r) = max p(r') for r'>r is an interpolation for p(r). '''

        ap = 0

        for i in range(re.size(0)):
            rn1 = re[i]
            rn = re[i-1] if i>0 else 0
            
            p = torch.max(pr[i:])
            ap += (rn1 - rn)*p

        return ap

    def AP_11int(self, pr, re):
        '''Calculates Average Precision (AP) given precision and recall vectors. Uses VOC PASCAL 11 point interpolation (paper 2006) (ref https://github.com/rafaelpadilla/Object-Detection-Metrics#average-precision).
        AP = sum[p'(r)] for r in (0, 0.1, 0.2, ..., 1). 
        p'(r) is interpolation of p(r) given by p'(r) = max p(r') for r'>r ''' 

        ap = 0

        for r in torch.arange(0, 1.1, 0.1).to(pr.device):
            #Recall greater or equal than r
            idx = re >= r
            
            #Precision will be the max value for r'>r
            p = pr[idx]
            p = torch.max(p) if p.size(0) else 0

            #Sum into ap
            ap += p

        ap /= 11
        return ap

