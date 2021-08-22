import torch
import torch.nn as nn
from .torch_util import Conv2d
import logging
logger =logging.getLogger('global')

class NaiveRpnHead(nn.Module):
    def __init__(self, inplanes, num_classes, num_anchors):
        '''
        Args:
            inplanes: input channel
            num_classes: as the name implies
            num_anchors: as the name implies
        '''
        super(NaiveRpnHead, self).__init__()
        self.num_anchors, self.num_classes = num_anchors, num_classes
        self.conv3x3 = nn.Conv2d(inplanes, 512, kernel_size=3, stride=1, padding=1)
        self.relu3x3 = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(
            512, num_anchors * num_classes, kernel_size=1, stride=1)
        self.conv_loc = nn.Conv2d(
            512, num_anchors * 7, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        Args:
            x: [B, inplanes, h, w], input feature
        Return:
            pred_cls: [B, num_anchors, h, w]
            pred_loc: [B, num_anchors*4, h, w]
        '''
        x = self.conv3x3(x)
        x = self.relu3x3(x)
        pred_cls = self.conv_cls(x)
        pred_loc = self.conv_loc(x)
        return pred_cls, pred_loc

class RPN(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(RPN, self).__init__()
        self.conv1_1 = Conv2d(64*2, 128, 2, padding=(1, 1))
        self.conv1_2 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv1_3 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv1_4 = Conv2d(128, 128, 1, padding=(1, 1))

        self.conv2_1 = Conv2d(128, 128, 2, padding=(1, 1))
        self.conv2_2 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv2_2 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv2_3 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv2_4 = Conv2d(128, 128, 1, padding=(1, 1))
        self.conv2_5 = Conv2d(128, 128, 1, padding=(1, 1))

        self.conv3_1 = Conv2d(128, 256, 2, padding=(1, 1))
        self.conv3_2 = Conv2d(256, 256, 1, padding=(1, 1))
        self.conv3_3 = Conv2d(256, 256, 1, padding=(1, 1))
        self.conv3_3 = Conv2d(256, 256, 1, padding=(1, 1))
        self.conv3_4 = Conv2d(256, 256, 1, padding=(1, 1))
        self.conv3_5 = Conv2d(256, 256, 1, padding=(1, 1))

        # self.deconv1 = nn.ConvTranspose2d(256, 256, 4, 4, (0, 1), (0, 1))
        # self.deconv2 = nn.ConvTranspose2d(128, 256, 2, 2, (0, 1), (0, 1))
        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, 4, 0)
        self.deconv2 = nn.ConvTranspose2d(128, 256, 2, 2, 0)
        self.deconv3 = nn.ConvTranspose2d(128, 256, 3, 1, 1)

        self.rpn_head = NaiveRpnHead(256*3, num_classes, num_anchors)

    def forward(self, x):
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        conv1 = self.conv1_3(conv1)
        conv1 = self.conv1_4(conv1)

        conv2 = self.conv2_1(conv1)
        conv2 = self.conv2_2(conv2)
        conv2 = self.conv2_2(conv2)
        conv2 = self.conv2_3(conv2)
        conv2 = self.conv2_4(conv2)
        conv2 = self.conv2_5(conv2)

        conv3 = self.conv3_1(conv2)
        conv3 = self.conv3_2(conv3)
        conv3 = self.conv3_3(conv3)
        conv3 = self.conv3_3(conv3)
        conv3 = self.conv3_4(conv3)
        conv3 = self.conv3_5(conv3)

        deconv1 = self.deconv1(conv3)
        deconv2 = self.deconv2(conv2)
        deconv3 = self.deconv3(conv1)
        logger.debug('x shape: {}'.format(x.size()))
        logger.debug('rpn conv1 shape:{}, conv2 shape:{}, conv3 shape:{}'.format(conv1.size(), conv2.size(),conv3.size()))
        logger.debug('deconv1 shape: {}'.format(deconv1.size()))
        logger.debug('deconv2 shape: {}'.format(deconv2.size()))
        logger.debug('deconv3 shape: {}'.format(deconv3.size()))
        out = torch.cat((deconv1, deconv2, deconv3), 1) ###################
        rpn_pred_cls, rpn_pred_loc = self.rpn_head(out)
        return rpn_pred_cls, rpn_pred_loc