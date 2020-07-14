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
from symbol import resnest
import time


def embedding_2_class_level_loss(embedding, loss_type, gt_label, batch_size):
    #embedding = mx.sym.BatchNorm(data=embedding, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_ebd')
    dataType = np.float32

    # label smoothing
    smooth_alpha = 0
    if config.label_smoothing:
        smooth_alpha = 0.1
    if config.fp_16:
        #all_label = mx.sym.Cast(data=all_label, dtype=np.float16)
        dataType = np.float16
    if loss_type=='softmax': #softmax
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
            lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        if config.fc7_no_bias:
            fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        else:
            _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
            fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=config.num_classes, name='fc7')
        orgLogits = fc7

    elif loss_type=='margin_softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
            lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        s = config.loss_s
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        cosData = fc7 / s
        #orgLogits = cosData
        if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
            if config.loss_m1==1.0 and config.loss_m2==0.0:
                s_m = s*config.loss_m3
                gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = s_m, off_value = 0.0, dtype=dataType)
                fc7 = fc7-gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy/s
                t = mx.sym.arccos(cos_t)
                if config.loss_m1!=1.0:
                    t = t*config.loss_m1
                if config.loss_m2>0.0:
                    t = t+config.loss_m2
                body = mx.sym.cos(t)
                if config.loss_m3>0.0:
                    body = body - config.loss_m3
                new_zy = body*s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0, dtype=dataType)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7+body
        orgLogits = fc7

    elif loss_type == 'svx_softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
            lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        s = config.loss_s
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        cosData = fc7
        #orgLogits = cosData
        if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
            if config.loss_m1==1.0 and config.loss_m2==0.0:
                gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = config.loss_m3, off_value = 0.0, dtype=dataType)
                fc7 = fc7-gt_one_hot
            else:
                cos_t = mx.sym.pick(fc7, gt_label, axis=1)
                t0 = mx.sym.arccos(cos_t)
                t = mx.sym.arccos(cos_t)
                if config.loss_m1!=1.0:
                    t = t*config.loss_m1
                if config.loss_m2>0.0:
                    t = t+config.loss_m2
                # 0<=theta+m<=pi
                #piSymbol = mx.sym.Variable(name='PI', shape=(1, ), lr_mult=0, init=mx.init.Constant(3.1415926))
                #piSymbol = mx.sym.ones((1))*3.1415926
                #thMask = mx.sym.broadcast_greater_equal(t, piSymbol)
                #fixed_param_names.append("PI")
                #t = mx.sym.where(thMask, t0, t) # boundary protect
                body = mx.sym.cos(t)
                #sin_m = math.sin(m)
                #mm = sin_m * m
                #keep_val = s*(cos_t - mm)   #tricks : additive margin instead
                if config.loss_m3>0.0:
                    body = body - config.loss_m3
                body = mx.sym.expand_dims(body, 1)

            # inter class  
            ## mask selection
            cosTheta = fc7
            nonGroundTruthMask = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 0.0, off_value = 1.0, dtype=dataType)

            ### add margin for non groundth class
            if config.loss_nm1 > 1.0 or config.loss_nm2 > 0.0 or config.loss_nm3 > 0.0:
                nt0 = mx.sym.arccos(cosTheta)
                nt = mx.sym.arccos(cosTheta)
                if config.loss_nm1!=1.0:
                    nt = nt/config.loss_nm1
                if config.loss_nm2>0.0:
                    nt = nt-config.loss_nm2
                # 0<=nm1*theta-nm2<=pi
                #zeroSymbol = mx.sym.Variable(name='ZERO', shape=(1, ), lr_mult=0, init=mx.init.Constant(0))
                #zeroSymbol = mx.sym.zeros((1))
                #nthMask = mx.sym.broadcast_lesser(nt, zeroSymbol)
                #fixed_param_names.append("ZERO")
                #nt = mx.sym.where(nthMask, nt0, nt)
                cosTheta = mx.sym.cos(nt)
                if config.loss_nm3>0.0:
                    cosTheta = cosTheta + config.loss_nm3
            hardMask = mx.sym.broadcast_greater_equal(cosData, body)*nonGroundTruthMask # use cosData instead of cosTheta for difficulty lowering 
            ## calculation of interCosTheta
            interCosTheta = ((config.mask - 1) * cosTheta + config.mask - 1.0)*hardMask + cosTheta*nonGroundTruthMask

            # intra class        
            gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0, dtype=dataType)
            intraCosTheta = mx.sym.broadcast_mul(gt_one_hot, body)
            fc7 = interCosTheta + intraCosTheta
        fc7 = fc7 * s
        orgLogits = fc7

    elif loss_type == 'softmax_circle_loss':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
            lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        #out_list.append(mx.symbol.BlockGrad(_weight))
        gamma = config.loss_gamma
        margin = config.loss_margin
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        fc_org = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        fc_org = fc_org / 2 + 0.5
        #orgLogits = fc_org
        Op =  1 + margin
        On = margin * -1
        delta_p = 1 - margin
        delta_n = margin
        gt_one_hot_p = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0, dtype=dataType)
        gt_one_hot_n = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 0.0, off_value = 1.0, dtype=dataType)
        delta_p_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = delta_p, off_value = 0.0, dtype=dataType)
        delta_n_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 0.0, off_value = delta_n, dtype=dataType)
        Op_tensor =  mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = Op, off_value = 0.0, dtype=dataType)
        On_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 0.0, off_value = On, dtype=dataType)

        fc_p = mx.sym.broadcast_mul(fc_org, gt_one_hot_p)
        a_p_tensor = Op_tensor - fc_p
        a_p_tensor = mx.sym.Activation(data = a_p_tensor, act_type = 'relu')
        sub_delta_p = fc_p - delta_p_tensor
        fc_p = mx.sym.broadcast_mul(a_p_tensor, sub_delta_p)

        fc_n = mx.sym.broadcast_mul(fc_org, gt_one_hot_n)
        a_n_tensor =  fc_n - On_tensor
        a_n_tensor = mx.sym.Activation(data = a_n_tensor, act_type = 'relu')
        sub_delta_n = fc_n - delta_n_tensor
        fc_n = mx.sym.broadcast_mul(a_n_tensor, sub_delta_n)

        final_fc = fc_n + fc_p
        fc7 = final_fc * gamma
        orgLogits = gamma * (sub_delta_p + sub_delta_n)

    elif loss_type=='arc_circle_softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
                    lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        gamma = config.loss_gamma
        thetaMargin = config.thetaMargin
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        #embedding = embedding / 100.0 
        nembedding = mx.symbol.L2Normalization(data=embedding, mode='instance', name='fc1n')
        #nembedding = mx.sym.BlockGrad(nembedding)
        fc_org = mx.sym.FullyConnected(nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        #fc_org = mx.sym.BlockGrad(fc_org)
        angleTheta_org = mx.sym.arccos(fc_org)
        orgLogits = fc_org

        delta_p = thetaMargin
        delta_n = math.pi - thetaMargin
        thetaShift = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = delta_p, off_value = delta_n, dtype=dataType)
        #angleTheta = angleTheta + thetaShift
        #angleTheta = mx.sym.clip(angleTheta, 0, math.pi)
        #fc7 = mx.sym.cos(angleTheta)
        #fc7 = fc7 * gamma
        #print("thetaMargin~~~~~~~~~~~~~`", thetaMargin)
        O_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = delta_n, off_value = delta_p, dtype=dataType)
        A_coe_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1, off_value = -1, dtype=dataType)

        angleTheta = angleTheta_org + thetaShift
        angleTheta = mx.sym.clip(angleTheta, 0, math.pi)
        fc_margin = mx.sym.cos(angleTheta)
        angleTheta_reserve = mx.sym.broadcast_mul(angleTheta_org, A_coe_tensor)
        A_tensor = angleTheta_reserve + O_tensor
        A_tensor = mx.sym.Activation(data = A_tensor, act_type = 'relu')
        final_fc = mx.sym.broadcast_mul(fc_margin, A_tensor)# + A_coe_tensor * math.pi * 0.1
        #only_positive_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1, off_value = 0, dtype=dataType)
        #final_fc = mx.sym.broadcast_mul(final_fc, only_positive_tensor)
        fc7 = final_fc * gamma
        #orgLogits = gamma * fc_margin
    elif loss_type=='fix_arc_circle_softmax':
        _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),
                    lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
        gamma = config.loss_gamma
        thetaMargin = config.thetaMargin
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        fc_org = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
        angleTheta_org = mx.sym.arccos(fc_org)
        orgLogits = fc_org

        delta_p = thetaMargin
        delta_n = -1 * thetaMargin

        O_p = math.pi - thetaMargin
        O_n = math.pi * 2 - thetaMargin

        thetaShift = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = delta_p, off_value = delta_n, dtype=dataType)
        #angleTheta = angleTheta + thetaShift
        #angleTheta = mx.sym.clip(angleTheta, 0, math.pi)
        #fc7 = mx.sym.cos(angleTheta)
        #fc7 = fc7 * gamma
        #print("thetaMargin~~~~~~~~~~~~~`", thetaMargin)
        O_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = O_p, off_value = O_n, dtype=dataType)
        A_coe_tensor = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1, off_value = -1, dtype=dataType)

        angleTheta = angleTheta_org + thetaShift
        angleTheta = mx.sym.clip(angleTheta, 0, math.pi)
        fc_margin = mx.sym.cos(angleTheta)
        angleTheta_reserve = mx.sym.broadcast_mul(angleTheta_org, A_coe_tensor)
        A_tensor = angleTheta_reserve + O_tensor
        A_tensor = mx.sym.Activation(data = A_tensor, act_type = 'relu')
        final_fc = mx.sym.broadcast_mul(fc_margin, A_tensor) + 0.1 * (math.pi * 2 - thetaMargin * 2) * A_coe_tensor
        fc7 = final_fc * gamma





    gradScale = 1
    if config.fp_16:
        fc7 = mx.sym.Cast(data=fc7, dtype=np.float32)
        orgLogits = mx.sym.Cast(data=orgLogits, dtype=np.float32)
        gradScale = config.scale16
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid', smooth_alpha=smooth_alpha, grad_scale=gradScale)
        #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
    body = mx.symbol.SoftmaxActivation(data=fc7)
    body = mx.symbol.log(body+1e-28)
    _label = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)
    body = body*_label
    ce_loss = mx.symbol.sum(body)/batch_size

    return softmax, orgLogits, ce_loss

