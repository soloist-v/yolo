import torch
from torch import nn


class VideoLoss(nn.Module):
    def __init__(self, ):
        super(VideoLoss, self).__init__()
        self.loss_cls = nn.BCELoss()

    def __call__(self, pred, targets):
        # pred: B, N
        # dtype = pred.dtype
        # device = pred.device
        bs = len(pred)
        l_cls = self.loss_cls(pred, targets)
        return l_cls / bs
