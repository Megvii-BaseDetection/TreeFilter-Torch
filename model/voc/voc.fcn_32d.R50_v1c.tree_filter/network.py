# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50
from seg_opr.seg_oprs import ConvBnRelu
from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

class Network(nn.Module):
    def __init__(self, out_planes, criterion, inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(Network, self).__init__()
        business_channel_num = config.business_channel_num
        embed_channel_num = config.embed_channel_num

        self.backbone = resnet50(pretrained_model, inplace=inplace,
                                 norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64)

        self.latent_layers = nn.ModuleList()
        self.refine_layers = nn.ModuleList()
        self.embed_layers = nn.ModuleList()
        self.mst_layers = nn.ModuleList()
        self.tree_filter_layers = nn.ModuleList()
        self.predict_layer = PredictHead(business_channel_num, out_planes, 4, norm_layer=norm_layer)
        for idx, channel in enumerate(self.backbone.layer_channel_nums[::-1]):
            self.latent_layers.append(
                ConvBnRelu(channel, business_channel_num, 3, 1, 1, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))
            self.refine_layers.append(
                ConvBnRelu(business_channel_num, business_channel_num, 1, 1, 0, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))
            self.embed_layers.append(
                ConvBnRelu(business_channel_num, embed_channel_num, 1, 1, 0, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))
            self.mst_layers.append(MinimumSpanningTree(TreeFilter2D.norm2_distance))
            self.tree_filter_layers.append(TreeFilter2D(groups=16))

        self.business_layers = [self.latent_layers, self.refine_layers, self.predict_layer,
                                self.mst_layers, self.tree_filter_layers, self.embed_layers]

        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)
        blocks.reverse()

        last_fm = None
        refined_fms = []
        for idx, (fm, latent_layer, refine_layer,
                  embed_layer, mst, tree_filter) in enumerate(
            zip(blocks,
                self.latent_layers,
                self.refine_layers,
                self.embed_layers,
                self.mst_layers,
                self.tree_filter_layers)):
            latent = latent_layer(fm)
            if last_fm is not None:
                tree = mst(latent)
                embed = embed_layer(last_fm)
                fusion = latent + tree_filter(last_fm, embed, tree)
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
