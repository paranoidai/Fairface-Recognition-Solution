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
from pair_wise_loss import embedding_2_pairwise_loss
from class_level_loss import embedding_2_class_level_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None
fixed_param_names = []


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--dataset', default=default.dataset, help='dataset config')
    parser.add_argument('--network', default=default.network, help='network config')
    parser.add_argument('--loss', default=default.loss, help='loss config')
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    
    # custom
    parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
    parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
    parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
    parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
    parser.add_argument('--num-workers', type=int, default=default.num_workers, help='number of workers for data loading')
    parser.add_argument('--cos-lr', action='store_true', help='whether to use cosine lr schedule.')
    parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
    parser.add_argument('--end-epoch', type=int, default=default.end_epoch, help='number of training epochs (default: 120)')
    parser.add_argument('--frequent', type=int, default=default.frequent, help='Number of batches to wait before logging.')
    parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
    parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
    parser.add_argument('--opt', type=str, default=default.opt, help='optmizer name')
    parser.add_argument('--no-wd', action='store_true', help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--selected-attributes', type=int,default=None)
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--freeze-block', type = int, default = 0,
                            help='whether to freeze the pre-layer for finetune')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--model-visual', action='store_true',
                        help='visualize Neural Networks as computation graph.')
    
    args = parser.parse_args()
    return args


def get_symbol(args):
    if(config.net_output == 'ECCV'):
        embedding,attr_softmax,body = eval(config.net_name).get_symbol(fixed_param_names=fixed_param_names)


        all_label = mx.symbol.Variable('softmax_label')
        gt_label = all_label
        #class_label = mx.symbol.Variable('face_attr_label')
        #gt_label = class_label
        gt_label = mx.symbol.slice_axis(all_label, axis=1, begin=0, end=1)
        class_label = mx.symbol.slice_axis(all_label, axis=1, begin=1, end=2)
        gt_label = mx.symbol.Reshape(data = gt_label, shape = (-1))
        class_label = mx.symbol.Reshape(data = class_label, shape = (-1))
        #attr_softmax = mx.sym.FullyConnected(data=attr_softmax, num_hidden=4)
        #if(config.fp_16):
        #    attr_softmax = mx.sym.Cast(data=attr_softmax,  dtype=np.float32)
        #softmax_class = mx.symbol.SoftmaxOutput(data=attr_softmax, label = class_label, name='softmax', normalization='valid', grad_scale=128.0)

        attr_softmax_loss = mx.symbol.log(attr_softmax+1e-5)
        _label = mx.sym.one_hot(class_label, depth = 4, on_value = -1.0, off_value = 0.0)
        attr_softmax_loss = attr_softmax_loss*_label
        attr_softmax_loss = mx.symbol.sum(attr_softmax_loss)/args.per_batch_size
        if(config.fp_16):
            attr_softmax_loss = mx.symbol.MakeLoss(attr_softmax_loss, grad_scale=config.scale16)
        else:
            attr_softmax_loss = mx.symbol.MakeLoss(attr_softmax_loss)
    else:

        embedding = eval(config.net_name).get_symbol(fixed_param_names=fixed_param_names)
        all_label = mx.symbol.Variable('softmax_label')
        gt_label = all_label


    

    #gt_label = class_label
    
    
    out_list = []
    out_list.append(mx.symbol.BlockGrad(embedding))
    if config.loss_name.find('fusion')>=0:
        triplet_loss_type = config.loss_name.split('_fusion_')[0]
        class_level_loss_type = config.loss_name.split('_fusion_')[1]
        print(triplet_loss_type, class_level_loss_type)
        triplet_loss = embedding_2_pairwise_loss(embedding, triplet_loss_type, gt_label, args.per_batch_size // 4 * 3)
        class_level_loss, orgLogits, ce_loss = embedding_2_class_level_loss(embedding, class_level_loss_type, gt_label, args.per_batch_size)
        out_list.append(mx.sym.BlockGrad(gt_label))
        out_list.append(triplet_loss)
        out_list.append(class_level_loss)
        final_loss = triplet_loss + ce_loss
        #out_list.append(mx.sym.BlockGrad(sp))
        #out_list.append(mx.sym.BlockGrad(sn))
        out_list.append(mx.sym.BlockGrad(final_loss))

    elif config.loss_name.find('triplet') >=0:
        triplet_batch_size = args.per_batch_size
        triplet_loss_type = config.loss_name
        print('triplet_loss_type ', triplet_loss_type)
        triplet_loss = embedding_2_pairwise_loss(embedding, triplet_loss_type, gt_label, triplet_batch_size)
        out_list.append(mx.sym.BlockGrad(gt_label))
        #out_list.append(mx.sym.BlockGrad(sp))
        #out_list.append(mx.sym.BlockGrad(sn))
        out_list.append(triplet_loss)

    elif config.loss_name == 'final_softmax':
        
        anchor_index = mx.symbol.argmax(gt_label)
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        #anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=anchor_index)
        #ap = mx.sym.broadcast_mul(gt_one_hot, diff)
        gt_label = mx.sym.Reshape(data = gt_label, shape = (77, 1))
        ap_emb = mx.sym.broadcast_mul(nembedding,  gt_label)
        ap_emb = mx.symbol.sum(ap_emb, axis=0, keepdims=0)

        data_shape = {'data':(300,3,112,112)}
        ap_emb = mx.sym.Reshape(data = ap_emb, shape = (1, 512))
        ap = mx.sym.broadcast_mul(nembedding, ap_emb)
        ap = mx.symbol.sum(ap, axis=1, keepdims=1)
        loss = 1 - ap
        #arg_shape, out_shape, _ = ap_emb.infer_shape(**data_shape)
        #print(out_shape)
        #exit()


        #final_loss = ap + gt_label
        #final_loss = mx.sym.broadcast_add(ap, gt_label)
        #final_loss = mx.sym.sum(final_loss, axis=1, keepdims=1)
        triplet_loss = mx.symbol.MakeLoss(loss)
        out_list.append(mx.sym.BlockGrad(gt_label))
        out_list.append(triplet_loss)

    elif config.loss_name.find('softmax')>=0:
        class_level_loss_type = config.loss_name
        class_level_loss, orgLogits, ce_loss = embedding_2_class_level_loss(embedding, class_level_loss_type, gt_label, args.per_batch_size)
        out_list.append(mx.symbol.BlockGrad(mx.symbol.SoftmaxActivation(data=orgLogits)))
        
        if(config.net_output == 'ECCV'):
            out_list.append(attr_softmax_loss)
        out_list.append(mx.sym.BlockGrad(class_level_loss))
        #out_list.append(mx.symbol.BlockGrad(orgLogits))
        if(config.net_output == 'ECCV'):
            out_list.append(mx.symbol.BlockGrad(attr_softmax))
            out_list.append(class_label)
        out_list.append(mx.sym.BlockGrad(ce_loss))

    out = mx.symbol.Group(out_list)
    return out

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx)==0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))
    curTime = time.strftime("%Y%m%d%H%M%S", time.localtime())
    prefix = os.path.join(args.models_root, '%s-%s-%s-%s'%(curTime, args.network, args.loss, args.dataset), 'model')
    prefix_dir = os.path.dirname(prefix)
    print('prefix', prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    args.ctx_num = len(ctx)
    args.batch_size = args.per_batch_size*args.ctx_num
    args.image_channel = config.image_shape[2]
    config.batch_size = args.batch_size
    config.per_batch_size = args.per_batch_size
    config.no_wd = args.no_wd
    config.last_gamma = args.last_gamma
    if(args.freeze_block == 1):
        config.bn_mom = 1.0


    print('bbbbbbbbbbbbbbbbbn', config.bn_mom)

    data_dir = config.dataset_path
    path_imgrec = None
    path_imglist = None
    image_size = config.image_shape[0:2]
    assert len(image_size)==2
    #assert image_size[0]==image_size[1]
    print('image_size', image_size)
    print('num_classes', config.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    print('Called with argument:', args, config)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    if len(args.pretrained)==0:
        arg_params = None
        aux_params = None
        sym = get_symbol(args)
    else:
        print('loading', args.pretrained, args.pretrained_epoch)
        _, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
        #for item in arg_params:
        #    print(item)
        #print(arg_params)
        #exit()
        sym = get_symbol(args)

    if args.model_visual:
        mx.viz.plot_network(sym,title='model',save_format='pdf',shape={'data':(64,3,224,224), 'label':(64,)}).view()
        exit(0)

    if config.count_flops:
        all_layers = sym.get_internals()
        pre_fix = ''
        if(config.emb_size == 2048):
            pre_fix = '2048_'
        _sym = all_layers[pre_fix + 'fc1_output']
        FLOPs = flops_counter.count_flops(_sym, data=(1,3,image_size[0],image_size[1]))
        _str = flops_counter.flops_str(FLOPs)
        print('Network FLOPs: %s'%_str)

    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    emb_symbol = sym.get_internals()[pre_fix + 'fc1_output']
    fixed_param_names = []
    if(args.freeze_block == 1):
        fixed_param_names = emb_symbol.list_arguments()
    elif(args.freeze_block == 2):
        emb_symbol = sym.get_internals()[pre_fix + 'bn1_output']
        fixed_param_names = emb_symbol.list_arguments()
    print(fixed_param_names)
    #fixed_aux = emb_symbol.list_auxiliary_states()
    #fixed_param_names.extend(fixed_aux)
    #print('ffffffffffffffixed params : ', fixed_param_names)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
        fixed_param_names = fixed_param_names
    )
    val_dataiter = None

    if config.loss_name.find('fusion')>=0:
        from pair_fusion_class_image_iter import FaceImageIter
        triplet_params = [config.triplet_bag_size, config.triplet_alpha, config.triplet_max_ap]
        train_dataiter = FaceImageIter(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = path_imgrec,
            shuffle              = True,
            rand_mirror          = config.data_rand_mirror,
            mean                 = mean,
            cutoff               = config.data_cutoff,
            ctx_num              = args.ctx_num,
            images_per_identity  = config.images_per_identity,
            triplet_params       = triplet_params,
            mx_model             = model,
            fairface_mode        = config.fairface_mode,
        )
        _metric = LossValueMetric()
        eval_metrics = [mx.metric.create(_metric)]


    elif config.loss_name.find('triplet')>=0:
        #from fair_face_triplet_iter import FaceImageIter
        from triplet_image_iter import FaceImageIter
        if(config.loss_name == 'triplet'):
            dis_type = 'e'
        elif(config.loss_name == 'atriplet'):
            dis_type = 'c'
        triplet_params = [config.triplet_bag_size, config.triplet_alpha, config.triplet_max_ap]
        train_dataiter = FaceImageIter(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = path_imgrec,
            shuffle              = True,
            rand_mirror          = config.data_rand_mirror,
            mean                 = mean,
            cutoff               = config.data_cutoff,
            ctx_num              = args.ctx_num,
            images_per_identity  = config.images_per_identity,
            triplet_params       = triplet_params,
            mx_model             = model,
            fairface_mode        = config.fairface_mode,
            dis_type             = dis_type,
        )
        _metric = LossValueMetric()
        eval_metrics = [mx.metric.create(_metric)]
        
    elif config.loss_name.find('softmax')>=0:
        from image_iter_gluon import FaceImageDataset
        train_dataset = FaceImageDataset(
          batch_size           = args.batch_size,
          data_shape           = data_shape,
          path_imgrec          = path_imgrec,
          shuffle              = True,
          rand_mirror          = config.data_rand_mirror,
          mean                 = mean,
          cutoff               = config.data_cutoff,
          color_jittering      = config.data_color,
          images_filter        = config.data_images_filter,
          selected_attributes  = args.selected_attributes,
          label_name           = ['softmax_label']
        )
        
        train_data = mx.gluon.data.DataLoader(train_dataset, args.batch_size, shuffle=True, last_batch="rollover", num_workers=args.num_workers)
        train_dataiter = mx.contrib.io.DataLoaderIter(train_data)
        
        metric1 = AccMetric()
        eval_metrics = [mx.metric.create(metric1)]
        if config.ce_loss:
            metric2 = LossValueMetric()
            eval_metrics.append( mx.metric.create(metric2) )
    else:
        from image_iter import FaceImageIter
        train_dataiter = FaceImageIter(
            batch_size           = args.batch_size,
            data_shape           = data_shape,
            path_imgrec          = path_imgrec,
            shuffle              = True,
            rand_mirror          = config.data_rand_mirror,
            mean                 = mean,
            cutoff               = config.data_cutoff,
            color_jittering      = config.data_color,
            images_filter        = config.data_images_filter,
        )
        
        metric1 = AccMetric()
        eval_metrics = [mx.metric.create(metric1)]
    if config.loss_name == 'final_softmax':
        _metric = LossValueMetric()
        eval_metrics = [mx.metric.create(_metric)]

        if config.ce_loss:
            metric2 = LossValueMetric()
            eval_metrics.append( mx.metric.create(metric2) )
    
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    #initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0 / args.ctx_num
    clip_gradient = None
    if config.fp_16:
        _rescale /= config.scale16
        clip_gradient = config.gradThres
    #opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)#, multi_precision=config.fp_16)
    opt = optimizer.create(args.opt, learning_rate=args.lr, momentum=config.mom, wd=config.wd, rescale_grad=_rescale, multi_precision=config.fp_16, clip_gradient=clip_gradient)
    _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

    # cos learning rate scheduler
    if args.cos_lr:
        num_batches = config.num_training_samples // args.batch_size
        total_batches = default.end_epoch * num_batches
      
    ver_list = []
    ver_name_list = []
    for name in config.val_targets:
        path = os.path.join(data_dir,name+".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)
            print('ver', name)

    def ver_test(nbatch):
        results = []
        label_shape = None
        if(config.net_output == 'ECCV'):
            label_shape = (args.batch_size, 2)
        
        for i in range(len(ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, args.batch_size, 10, None, label_shape)
            print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
            #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
            results.append(acc2)
        return results

    highest_acc = [0.0, 0.0]  #lfw and target
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    highestStep = [0]
    lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
        #global global_step
        global_step[0]+=1
        mbatch = global_step[0]
        
        if config.useWarmup and (mbatch < config.warmupSteps):
            #opt.lr = args.lr * mbatch / config.warmupSteps
            opt.lr = 1.0e-8
            #print("warmup lr: ", opt.lr)
        
        if (not config.useWarmup) or (config.useWarmup and (mbatch >= config.warmupSteps)):
            targetSteps = mbatch
            if config.useWarmup:
                if mbatch==config.warmupSteps:
                    opt.lr = args.lr
            
                targetSteps -= config.warmupSteps
            
            if args.cos_lr:
                opt.lr  = 0.5 * args.lr * (1 + np.cos(np.pi * (targetSteps / total_batches)))
                if (targetSteps % 500) == 0:
                    print('cos lr change to', opt.lr)
            else:
                for step in lr_steps:
                    if targetSteps==step:
                        opt.lr *= 0.1
                        print('lr change to', opt.lr)
                        break

        _cb(param)
        if mbatch%1000==0:
            print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

        if mbatch>=0 and mbatch%args.verbose==0:
            acc_list = ver_test(mbatch)
            save_step[0]+=1
            msave = save_step[0]
            do_save = False
            is_highest = False
            if len(acc_list)>0:
                score = sum(acc_list)
                if acc_list[-1]>=highest_acc[-1]:
                    if acc_list[-1]>highest_acc[-1]:
                        is_highest = True
                    else:
                        if score>=highest_acc[0]:
                            is_highest = True
                            highest_acc[0] = score
                    highest_acc[-1] = acc_list[-1]
                    highestStep[0] = save_step[0]
                    
            if is_highest:
                do_save = True
            if args.ckpt==0:
                do_save = False
            elif args.ckpt==2:
                do_save = True
            elif args.ckpt==3:
                msave = 1

            if do_save:
                print('saving', msave)
                arg, aux = model.get_params()
                if config.ckpt_embedding:
                    all_layers = model.symbol.get_internals()
                    _sym = all_layers['fc1_output']
                    _arg = {}
                    for k in arg:
                        if not k.startswith('fc7'):
                            _arg[k] = arg[k]
                    mx.model.save_checkpoint(prefix, msave, _sym, _arg, aux)
                else:
                    mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            print('[%d]Accuracy-Highest: %1.5f, mbatch: %d'%(mbatch, highest_acc[-1], highestStep[0]))
        if config.max_steps>0 and mbatch>config.max_steps:
            sys.exit(0)

    epoch_cb = None
    if config.loss_name.find('triplet') < 0:
        train_dataiter = mx.io.PrefetchingIter(train_dataiter)  #triplet loss unavailable
    ######
    if(config.net_output == 'ECCV'):
        class_metric = AccMetric(acc_name = 'class_acc', label_index = 1, pred_index = 4)
        eval_metrics.append(mx.metric.create(class_metric))

        eval_metrics,
    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = 999999,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = args.kvstore,
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

