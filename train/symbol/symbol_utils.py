import sys
import os
import mxnet as mx
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import config, default
import numpy as np
from mxnet.gluon import nn

swish_index = 0

def gluon_act(act_type):
    if act_type=='prelu':
        return nn.PReLU()
    else:
        return nn.Activation(act_type)

def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name = 'act', lr_mult = 1.0):
    if act_type=='prelu':
        body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name, lr_mult = lr_mult)
    elif act_type == 'swish':
        tmp_sigmoid = mx.symbol.Activation(data=data, act_type='sigmoid', name=name + '_sigmoid')
        global swish_index
        body = mx.symbol.elemwise_mul(data, tmp_sigmoid, name= 'swish_' + str(swish_index))
        swish_index += 1
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)

    return body

bn_mom = config.bn_mom
def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bghted_fc1n = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
    return bn

def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
    body = last_conv
    if fc_type=='Z':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = body
    elif fc_type=='E':
        pre_fix = ''
        if(config.emb_size == 2048):
            pre_fix = '2048_'
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=pre_fix + 'bn1')
        #print('eeeeeeeeeeeeeeeee')
        #body.list_attr()
        body = mx.symbol.Dropout(data=body, p=config.finalDrop)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name=pre_fix + 'pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name=pre_fix+'fc1')

    elif fc_type == 'ECCV':
        
        class_branch_unit_count = 4
        
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        body = mx.symbol.Dropout(data=body, p=config.finalDrop)

        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
        org_fc1 = fc1

        fc2 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc2')
        fc2 = mx.sym.BatchNorm(data=fc2, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc2')
        fc2 = mx.sym.Reshape(data = fc2, shape = (-1, num_classes, 1))

        fc3 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc3')
        fc3 = mx.sym.BatchNorm(data=fc3, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc3')
        fc3 = mx.sym.Reshape(data = fc3, shape = (-1, num_classes, 1))

        fc4 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc4')
        fc4 = mx.sym.BatchNorm(data=fc4, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc4')
        fc4 = mx.sym.Reshape(data = fc4, shape = (-1, num_classes, 1))

        fc5 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc5')
        fc5 = mx.sym.BatchNorm(data=fc5, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc5')
        fc5 = mx.sym.Reshape(data = fc5, shape = (-1, num_classes, 1))

        #data_shape = {'data':(300,3,112,112)}
        branch_fc = mx.sym.concat(fc2, fc3, fc4, fc5, dim = 2)
        #arg_shape, out_shape, _ = branch_fc.infer_shape(**data_shape)
        #print(out_shape)
        #exit()

        



        #fc1 = mx.sym.Reshape(data = fc1, shape = (-1, 1, num_classes))
        #fc1 = mx.sym.FullyConnected(data= mx.symbol.BlockGrad(fc1), num_hidden=num_classes * 4, name='added_fc1')
        #fc1 = mx.sym.FullyConnected(data= mx.symbol.BlockGrad(org_fc1), num_hidden=num_classes * class_branch_unit_count, name='added_fc1')
        #fc1 = mx.sym.FullyConnected(data= org_fc1, num_hidden=num_classes * class_branch_unit_count, name='added_fc1')
        #fc1 = mx.sym.broadcast_axis(data = fc1, axis=1, size=4)
        #fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
        class_branch = mx.sym.FullyConnected(data=mx.symbol.BlockGrad(body), num_hidden=num_classes, name='class_branch')
        #class_branch = mx.sym.FullyConnected(data = body, num_hidden=num_classes, name='class_branch')
        class_branch = mx.sym.BatchNorm(data=class_branch, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='class_branch_bn1')
        class_branch = mx.sym.FullyConnected(data=org_fc1, num_hidden = class_branch_unit_count, name ='class_branch_fc')
        #class_branch = class_branch / 16.0
        
        #class_branch = mx.sym.Reshape(data = class_branch, shape = (-1, 1, 4))
        
        if(config.fp_16):
            class_branch = mx.sym.Cast(data=class_branch, dtype=np.float32)
            branch_fc = mx.sym.Cast(data=branch_fc, dtype=np.float32)
            org_fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
        softmax = mx.symbol.SoftmaxActivation(data=class_branch, name = 'class_softmax')
        #mul_softmax = None
        mul_softmax = mx.symbol.SoftmaxActivation(data=class_branch)
        mul_softmax = mx.sym.Reshape(data = mul_softmax, shape = (-1, 1, class_branch_unit_count))
        #@fc1 = mx.sym.Reshape(data = fc1, shape = (-1, class_branch_unit_count, num_classes))

        #@if(config.fp_16):
        #@    mul_softmax = mx.sym.Cast(data=mul_softmax, dtype=np.float16)
        

        #data_shape = {'data':(300,3,112,112)}
        fc1 = mx.sym.broadcast_mul(mx.symbol.BlockGrad(mul_softmax), branch_fc, name = 'mul_fc1')
        #fc1 = mx.sym.broadcast_mul(mul_softmax, fc1, name = 'mul_fc1')
        #arg_shape, out_shape, _ = fc1.infer_shape(**data_shape)

        fc1 = mx.sym.mean(data=fc1, axis = 2, name = 'weighted_fc1')
        #arg_shape, out_shape, _ = fc1.infer_shape(**data_shape)
        fc1 = mx.sym.broadcast_add(fc1 , org_fc1, name = 'fusion_fc1')
        #fc1 = mx.sym.broadcast_add(fc1 , mx.symbol.BlockGrad(org_fc1), name = 'fusion_fc1')
        #arg_shape, out_shape, _ = fc1.infer_shape(**data_shape)

        #print(out_shape)
        #exit()

        if(config.fp_16):
            fc1 = mx.sym.Cast(data=fc1, dtype=np.float16)
        #class_branch = mx.sym.Reshape(data = softmax, shape = (-1, 4))
        return fc1, softmax, mul_softmax

    elif fc_type == 'GEM':

        bn_body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        pk = mx.symbol.ones(shape = (1,))
        if config.fp_16:
          #pk = mx.symbol.Cast(data = pk, dtype = np.float16)
          org_body = mx.symbol.Cast(data = bn_body, dtype = np.float32)
        else:
          org_body = bn_body
        pk = pk * 3.0
        eps = 1e-4
        org_body = mx.symbol.broadcast_power(org_body, pk)
        pk_1 = 1.0 / pk
        pool1 = mx.symbol.Pooling(data = org_body, pool_type = 'avg', global_pool = True)
        #pool1 = pool1 + eps
        abs_pool1 = mx.symbol.abs(pool1)
        sign_tag = mx.symbol.broadcast_equal(pool1, abs_pool1) + -1 * mx.symbol.broadcast_not_equal(pool1, abs_pool1)
        abs_pool1 = abs_pool1 + eps


        #pool1 = pool1 + eps

        #abs_pool1 = abs_pool1 + eps
        pool2 = mx.symbol.broadcast_power(abs_pool1, pk_1)
        pool3 = mx.symbol.elemwise_mul(pool2, sign_tag)

        #data_shape = {'data':(300,3,112,112)}
        #arg_shape, out_shape, _ = body.infer_shape(**data_shape)
        #print('out_shape,', out_shape)
        #exit()

        #body = mx.symbol.Dropout(data=pool1, p=config.finalDrop)
        #fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.Flatten(data=pool3)
        if config.fp_16:
          fc1 = mx.symbol.Cast(data = fc1, dtype = np.float16)
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')

        return fc1#, pool1, pool3

    elif fc_type=='FC':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    elif fc_type=='SFC':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        body = Conv(data=body, num_filter=input_channel, kernel=(3,3), stride=(2,2), pad=(1,1),
                                no_bias=True, name="convf", num_group = input_channel)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnf')
        body = Act(data=body, act_type=config.net_act, name='reluf')
        body = Conv(data=body, num_filter=input_channel, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="convf2")
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bnf2')
        body = Act(data=body, act_type=config.net_act, name='reluf2')
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    elif fc_type=='GAP':
        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        relu1 = Act(data=bn1, act_type=config.net_act, name='relu1')
        # Although kernel is not used here when global_pool=True, we should put one
        pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
        flat = mx.sym.Flatten(data=pool1)
        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    elif fc_type=='GNAP': #mobilefacenet++
        filters_in = 512 # param in mobilefacenet
        if num_classes>filters_in:
            body = mx.sym.Convolution(data=last_conv, num_filter=num_classes, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, name='convx')
            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=0.9, name='convx_bn')
            body = Act(data=body, act_type=config.net_act, name='convx_relu')
            filters_in = num_classes
        else:
            body = last_conv
        body = mx.sym.BatchNorm(data=body, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn6f')  
        
        spatial_norm=body*body
        spatial_norm=mx.sym.sum(data=spatial_norm, axis=1, keepdims=True)
        spatial_sqrt=mx.sym.sqrt(spatial_norm)
        #spatial_mean=mx.sym.mean(spatial_sqrt, axis=(1,2,3), keepdims=True)
        spatial_mean=mx.sym.mean(spatial_sqrt)
        spatial_div_inverse=mx.sym.broadcast_div(spatial_mean, spatial_sqrt)
        
        spatial_attention_inverse=mx.symbol.tile(spatial_div_inverse, reps=(1,filters_in,1,1))   
        body=body*spatial_attention_inverse
        #body = mx.sym.broadcast_mul(body, spatial_div_inverse)
        
        fc1 = mx.sym.Pooling(body, kernel=(7, 7), global_pool=True, pool_type='avg')
        if num_classes<filters_in:
            fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn6w')
            fc1 = mx.sym.FullyConnected(data=fc1, num_hidden=num_classes, name='pre_fc1')
        else:
            fc1 = mx.sym.Flatten(data=fc1)
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=0.9, name='fc1')
    elif fc_type=="GDC": #mobilefacenet_v1
        conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(7,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")  
        #conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(4,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")  
        conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    elif fc_type=='F':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
    elif fc_type=='G':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
    elif fc_type=='H':
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='fc1')
    elif fc_type=='I':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    elif fc_type=='J':
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    return fc1

def antialiased_downsample(inputs, name, in_ch, fixed_param_names, pad_type='reflect', filt_size=3, stride=(2, 2), pad_off=0):
    pad_size = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
    pad_size = [x + pad_off for x in pad_size]

    def get_filter(filt_size, in_ch):
        if(filt_size==1):
            filter = np.array([1.,])
        elif(filt_size==2):
            filter = np.array([1., 1.])
        elif(filt_size==3):
            filter = np.array([1., 2., 1.])
        elif(filt_size==4):    
            filter = np.array([1., 3., 3., 1.])
        elif(filt_size==5):    
            filter = np.array([1., 4., 6., 4., 1.])
        elif(filt_size==6):    
            filter = np.array([1., 5., 10., 10., 5., 1.])
        elif(filt_size==7):    
            filter = np.array([1., 6., 15., 20., 15., 6., 1.])
        else:
            raise NotImplementedError('invalid filter size %d' % filt_size)
        filter = filter[None,:]*filter[:,None]
        filter = filter/filter.sum()
        filter = filter[None,None,:,:]
        filter = np.tile(filter,(in_ch,1,1,1))
        
        filter = filter.astype(np.float32)
        return filter
    
    W_val = get_filter(filt_size, in_ch)
    # padding
    inputs = mx.sym.pad(inputs, mode=pad_type, pad_width=(0,)*4+tuple(pad_size), name=name+"_padding", constant_value=0)
    # downsample
    blurPoolW = mx.sym.Variable(name+"_BlurPool_weight", shape=W_val.shape, init=mx.init.Constant(W_val))
    mx.sym.BlockGrad(blurPoolW)
    fixed_param_names.append(name+"_BlurPool_weight")
    out = mx.sym.Convolution(data=inputs, weight=blurPoolW, bias=None, no_bias=True, kernel=(filt_size,filt_size), num_filter=in_ch, num_group=in_ch, stride=stride, name=name+"_BlurPool")
    
    return out

def get_loc(data, attr={'lr_mult': '0.01'}):
    """
    the localisation network in stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    ## 与gluon写法一致，只是调用的mx.symbol模块
    loc = mx.sym.Convolution(data=data, num_filter=24, kernel=(5, 5), stride=(1, 1), name="stn_loc_conv1", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn1", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max', name="stn_loc_pool1")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act1", lr_mult=config.stn_fc1_lr_mult)

    loc = mx.sym.Convolution(data=loc, num_filter=48, kernel=(3, 3), stride=(1, 1), name="stn_loc_conv2", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn2", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max', name="stn_loc_pool2")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act2", lr_mult=config.stn_fc1_lr_mult)

    loc = mx.sym.Convolution(data=loc, num_filter=96, kernel=(3, 3), stride=(1, 1), name="stn_loc_conv3", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn3", lr_mult=config.stn_fc1_lr_mult)
    loc = mx.sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max', name="stn_loc_pool3")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act3", lr_mult=config.stn_fc1_lr_mult)

    _weight1 = mx.symbol.Variable("stn_loc_fc1_weight", shape=(64, 12*12*96),
                                  lr_mult=config.stn_fc1_lr_mult, wd_mult=config.stn_fc1_wd_mult, init=mx.init.Normal(0.01))
    loc = mx.sym.FullyConnected(data=loc, weight=_weight1, no_bias=True, num_hidden=64, name="stn_loc_fc1")
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn4")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act4", lr_mult=config.stn_fc1_lr_mult)

    _weight2 = mx.symbol.Variable("stn_loc_fc2_weight", shape=(64, 64),
                                  lr_mult=config.stn_fc2_lr_mult, wd_mult=config.stn_fc2_wd_mult, init=mx.init.Normal(0.01))
    loc = mx.sym.FullyConnected(data=loc, weight=_weight2, no_bias=True, num_hidden=64, name="stn_loc_fc2")
    loc = mx.sym.BatchNorm(data=loc, fix_gamma=False, eps=2e-5, momentum=config.bn_mom, name="stn_loc_bn5")
    loc = Act(data=loc, act_type=config.net_act, name="stn_loc_act5", lr_mult=config.stn_fc2_lr_mult)

    _weight3 = mx.symbol.Variable("stn_loc_fc3_weight", shape=(6, 64),
                                  lr_mult=config.stn_fc3_lr_mult, wd_mult=config.stn_fc3_wd_mult,
                                  init=mx.init.Normal(0.01))
    _bias3 = mx.symbol.Variable("stn_loc_fc3_bias", shape=(6, ),
                                lr_mult=config.stn_fc3_lr_mult, wd_mult=config.stn_fc3_wd_mult,
                                init=mx.init.Constant(mx.nd.array([1,0,0,0,1,0])))
    loc = mx.sym.FullyConnected(data=loc, weight=_weight3, bias=_bias3, num_hidden=6, name="stn_loc_fc3")
    return loc
