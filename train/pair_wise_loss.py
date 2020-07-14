from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import sklearn
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from config import config, default, generate_config
from metric import *
from common import flops_counter
from eval import verification
from symbol import fresnet
from symbol import fmobilefacenet
import time

def embedding_2_pairwise_loss(embedding, loss_type, gt_label, triplet_batch_size):
    if loss_type.find('triplet')>=0:
        if config.fp_16:
            embedding = mx.symbol.Cast(data = embedding, dtype = np.float32)

        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        #nembedding = embedding
        anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=triplet_batch_size//3)
        positive = mx.symbol.slice_axis(nembedding, axis=0, begin=triplet_batch_size//3, end=2*triplet_batch_size//3)
        negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2*triplet_batch_size//3, end=triplet_batch_size)
        if loss_type =='triplet':
            ap = anchor - positive
            an = anchor - negative
            ap = ap*ap
            an = an*an
            ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
            an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
            triplet_loss = mx.symbol.Activation(data = (ap-an+config.triplet_alpha), act_type='relu')
            triplet_loss = mx.symbol.mean(triplet_loss)

        elif loss_type == 'fix_arc_circle_triplet':
            print('fffffffffffffix_arc_circle_triplet')
            ap = anchor*positive
            an = anchor*negative
            ap = mx.symbol.sum(ap, axis=1, keepdims=1)
            an = mx.symbol.sum(an, axis=1, keepdims=1)
            thetaP = mx.sym.arccos(ap)
            thetaN = mx.sym.arccos(an)
            gamma = config.loss_gamma
            thetaMargin = config.thetaMargin

            delta_p = thetaMargin
            delta_n = 0 - thetaMargin

            Op = (math.pi - thetaMargin)
            On = (math.pi * 2 - thetaMargin)

            thetaP_m = thetaP + delta_p
            thetaP_m = mx.sym.clip(thetaP_m, 0, math.pi)
            thetaN_m = thetaN + delta_n
            thetaN_m = mx.sym.clip(thetaN_m, 0, math.pi)

            alphaP = thetaP + Op
            alphaP = mx.sym.Activation(data = alphaP, act_type = 'relu')
            alphaN = On - thetaN
            alphaN = mx.sym.Activation(data = alphaN, act_type = 'relu')

            sp = gamma * alphaP * mx.sym.cos(thetaP_m)
            sn = gamma * alphaN * mx.sym.cos(thetaN_m)

            triplet_loss = mx.symbol.Activation(data = (sn - sp), act_type='relu')
            triplet_loss = mx.symbol.mean(triplet_loss)
            triplet_loss = mx.symbol.MakeLoss(triplet_loss)
            if config.fp_16:
                triplet_loss = mx.symbol.MakeLoss(triplet_loss, grad_scale=config.scale16)


            #return triplet_loss, sp, sn


        elif loss_type == 'arc_circle_triplet':
            print('aaaaaaaaaaaaaaarc_circle_triplet')
            ap = anchor*positive
            an = anchor*negative
            ap = mx.symbol.sum(ap, axis=1, keepdims=1)
            an = mx.symbol.sum(an, axis=1, keepdims=1)
            thetaP = mx.sym.arccos(ap)
            thetaN = mx.sym.arccos(an)
            gamma = config.loss_gamma / math.pi 
            thetaMargin = config.thetaMargin

            delta_p = thetaMargin
            delta_n = math.pi - thetaMargin

            thetaP_m = thetaP + delta_p
            thetaP_m = mx.sym.clip(thetaP_m, 0, math.pi)
            thetaN_m = thetaN + delta_n
            thetaN_m = mx.sym.clip(thetaN_m, 0, math.pi)

            alphaP = thetaP + delta_n
            alphaP = mx.sym.Activation(data = alphaP, act_type = 'relu')
            alphaN = delta_p - thetaN
            alphaN = mx.sym.Activation(data = alphaN, act_type = 'relu')

            sp = gamma * alphaP * mx.sym.cos(thetaP_m)
            sn = gamma * alphaN * mx.sym.cos(thetaN_m)

            #n_loss = mx.sym.exp(sn)
            #n_loss = mx.symbol.sum(ap)
            #p_loss = mx.sym.exp(sp)
            #p_loss = mx.symbol.sum(sp)

            n_loss = mx.sym.exp(sn)
            n_loss = mx.symbol.sum(n_loss)
            p_loss = mx.sym.exp(sp * -1.0)
            p_loss = mx.symbol.sum(p_loss)

            #triplet_loss = mx.symbol.Activation(data = (sn - sp), act_type='relu')
            #triplet_loss = mx.symbol.mean(triplet_loss)


            triplet_loss = mx.sym.log(1 + n_loss * p_loss)
            triplet_loss = mx.symbol.MakeLoss(triplet_loss)
            if config.fp_16:
                triplet_loss = mx.symbol.MakeLoss(triplet_loss, grad_scale=config.scale16)

            #return triplet_loss, sp, sn


            
        else:
            ap = anchor*positive
            an = anchor*negative
            ap = mx.symbol.sum(ap, axis=1, keepdims=1) #(T,1)
            an = mx.symbol.sum(an, axis=1, keepdims=1) #(T,1)
            ap = mx.sym.arccos(ap)
            an = mx.sym.arccos(an)
            triplet_loss = mx.symbol.Activation(data = (ap-an+config.triplet_alpha), act_type='relu')
            triplet_loss = mx.symbol.mean(triplet_loss)
        if config.fp_16:
            triplet_loss = mx.symbol.MakeLoss(triplet_loss, grad_scale=config.scale16)
            #triplet_loss = mx.symbol.MakeLoss(triplet_loss)
        else:
            triplet_loss = mx.symbol.MakeLoss(triplet_loss)

    return triplet_loss
