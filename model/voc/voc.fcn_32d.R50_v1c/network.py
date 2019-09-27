# encoding: utf-8

import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50
from seg_opr.seg_oprs import ConvBnRelu

class Network(nn.Module):
    def __init__(self, out_planes, criterion, inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(Network, self).__init__()
        business_channel_num = config.business_channel_num

        self.backbone = resnet50(pretrained_model, inplace=inplace,
                                  norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)

        self.latent_layers = nn.ModuleList()
        self.refine_layers = nn.ModuleList()
        self.predict_layer = PredictHead(business_channel_num, out_planes, 4, norm_layer=norm_layer)
        for idx, channel in enumerate(self.backbone.layer_channel_nums[::-1]):
            self.latent_layers.append(
                ConvBnRelu(channel, business_channel_num, 3, 1, 1, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))
            self.refine_layers.append(
                ConvBnRelu(business_channel_num, business_channel_num, 1, 1, 0, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))

        self.business_layers = [self.latent_layers, self.refine_layers, self.predict_layer]

        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)
        blocks.reverse()

        last_fm = None
        refined_fms = []
        for idx, (fm, latent_layer, refine_layer) in enumerate(
                zip(blocks, self.latent_layers, self.refine_layers)):
            latent = latent_layer(fm)
            if last_fm is not None:
                fusion = latent + last_fm
                refined_fms.append(refine_layer(fusion))
            else:
                refined_fms.append(latent)
            last_fm = F.interpolate(refined_fms[-1], scale_factor=2, mode='bilinear',
                                    align_corners=True)

        pred = self.predict_layer(refined_fms[-1])

        if label is not None:
            loss = self.criterion(pred, label)
            return loss

        return pred


class PredictHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(PredictHead, self).__init__()
        self.head_layers = nn.Sequential(
            ConvBnRelu(in_planes, in_planes, 3, 1, 1, has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Conv2d(in_planes, out_planes, kernel_size=1,
                      stride=1, padding=0))
        self.scale = scale

    def forward(self, x):
        x = self.head_layers(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                          align_corners=True)
        return x
