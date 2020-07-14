##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import Conv2D, Block, HybridBlock, Dense, BatchNorm, Activation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config
import numpy as np


__all__ = ['SplitAttentionConv']

USE_BN = True

def gluon_act(act_type):
    if act_type=='prelu':
        return nn.PReLU()
    else:
        return nn.Activation(act_type)

class SplitAttentionConv(HybridBlock):
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, radix=2, *args, in_channels=None, r=2,
                 norm_layer=BatchNorm, norm_kwargs=None, drop_ratio=0, **kwargs):
        super().__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        inter_channels = max(in_channels*radix//2//r, 32)
        self.radix = radix
        self.cardinality = groups
        self.conv = Conv2D(channels*radix, kernel_size, strides, padding, dilation,
                           groups=groups*radix, *args, in_channels=in_channels, **kwargs)
        if USE_BN:
            self.bn = norm_layer(in_channels=channels*radix, **norm_kwargs)
        self.relu = gluon_act(config.net_act)
        self.fc1 = Conv2D(inter_channels, 1, in_channels=channels, groups=self.cardinality)
        if USE_BN:
            self.bn1 = norm_layer(in_channels=inter_channels, **norm_kwargs)
        self.relu1 = gluon_act(config.net_act)
        if drop_ratio > 0:
            self.drop = nn.Dropout(drop_ratio)
        else:
            self.drop = None
        self.fc2 = Conv2D(channels*radix, 1, in_channels=inter_channels, groups=self.cardinality)
        self.channels = channels
        self.rsoftmax = rSoftMax(radix, groups)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if USE_BN:
            x = self.bn(x)
        x = self.relu(x)

        if config.fp_16:
            x = F.cast(data=x, dtype='float32')

        if self.radix > 1:
            splited = F.split(x, self.radix, axis=1)            
            gap = sum(splited)
        else:
            gap = x
        gap = F.contrib.AdaptiveAvgPooling2D(gap, 1)
        if config.fp_16:
            gap = F.cast(data=gap, dtype='float16')
        gap = self.fc1(gap)
        if USE_BN:
            gap = self.bn1(gap)
        atten = self.relu1(gap)
        if self.drop:
            atten = self.drop(atten)
        atten = self.fc2(atten).reshape((0, self.radix, self.channels))
        if config.fp_16:
            atten = F.cast(data=atten, dtype='float32')
            self.rsoftmax.cast('float32')        
        atten = self.rsoftmax(atten).reshape((0, -1, 1, 1))
        if self.radix > 1:
            atten = F.split(atten, self.radix, axis=1)
            outs = [F.broadcast_mul(att, split) for (att, split) in zip(atten, splited)]
            out = sum(outs)
        else:
            out = F.broadcast_mul(atten, x)

        if config.fp_16:
            out = F.cast(data=out, dtype='float16')

        return out


class rSoftMax(nn.HybridBlock):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def hybrid_forward(self, F, x):
        if self.radix > 1:
            x = x.reshape((0, self.cardinality, self.radix, -1)).swapaxes(1, 2)
            #x = F.clip(F.softmax(x, axis=1), a_min=1e-9, a_max=1.0)
            x = F.softmax(x, axis=1, dtype='float32')
            x = x.reshape((0, -1))
        else:
            x = F.sigmoid(x, dtype='float32')
        return x

