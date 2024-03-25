import os
import random
import time
import math
import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class Genalize_dice_loss_multi_class_3D(nn.Module):
    def __init__(self, channel=7):
        super(Genalize_dice_loss_multi_class_3D, self).__init__()
        self.loss_lambda = [1, 2, 2, 2, 2, 2, 2]
        self.channel = channel

    def forward(self, logits, gt):
        dice = 0
        eps = 1e-7

        assert len(logits.shape) == 5, 'This loss is for 3D data (BCDHW), please check your output!'

        softmaxpred = logits

        for i in range(self.channel):
            inse = torch.sum(softmaxpred[:, i, :, :, :] * gt[:, i, :, :, :])
            l = torch.sum(softmaxpred[:, i, :, :, :])
            r = torch.sum(gt[:, i, :, :, :])
            dice += ((inse + eps) / (l + r + eps)) * self.loss_lambda[i] / sum(self.loss_lambda[:self.channel])

        return 1 - 2.0 * dice / self.channel


class Proto_Contrast(nn.Module):
    def __init__(self):
        super(Proto_Contrast, self).__init__()

        self.temperature = 0.7
        self.base_temperature = 0.7

    def _contrastive(self, feats_, labels_):
        '''
            feat_:  [N,256]
            labels_: [N]
        '''
        n_view = feats_.shape[0]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = feats_

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)

        logits = anchor_dot_contrast - logits_max.detach()

        # logits = anchor_dot_contrast
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
    def forward(self, feats, labels):
        '''
            feat:  [N,256]
            labels: [N]
        '''

        loss = self._contrastive(feats, labels)
        return loss

