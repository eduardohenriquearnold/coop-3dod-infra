import torch
import numpy as np

from lib.evaluator.iou import iou


def nms(pred, iou_threshold, score_threshold):
    '''Non maxima suppresion for predictions'''

    #filter by score threshold
    idx = pred[:,-1] > score_threshold
    pred = pred[idx]

    #order predictions by descending score
    idx = np.argsort(pred[:,-1])
    predScore = pred[idx, -1]
    pred = pred[idx, :-1]

    #contiguous array
    pred = np.ascontiguousarray(pred[::-1])
    predScore = np.ascontiguousarray(predScore[::-1])

    #calculate iou for each box
    for boxidx, box in enumerate(pred[:-1]):
        boxcopies = np.tile(box.reshape(1,-1), (len(pred)-boxidx-1,1))
        ious = iou(torch.FloatTensor(boxcopies), torch.FloatTensor(pred[boxidx+1:]))
        mask = (ious<iou_threshold).float().view(-1,1).repeat(1,7).numpy()
        pred[boxidx+1:] *= mask

    #remove zeroed entries
    idx = pred[:,3] > 0
    pred = pred[idx]
    predScore = predScore[idx]

    #Add scores back
    pred = np.hstack((pred, predScore.reshape(-1,1)))

    return pred


