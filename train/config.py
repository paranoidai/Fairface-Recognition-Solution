import numpy as np
import os
from easydict import EasyDict as edict
import math

# general
config = edict()
config.bn_mom = 0.9
config.wd = 0.0005
config.mom = 0.9
config.workspace = 64
config.emb_size = 512
#config.ckpt_embedding = True
config.ckpt_embedding = False
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = 0
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.fp_16 = False
config.fairface_mode = False
config.useWarmup = False
config.warmupSteps = 400
config.gradThres = None
config.scale16 = 128
config.finalDrop = 0.4
config.label_smoothing = False

# custom default settings
default = edict()
default.dataset = 'emore'
default.network = 'r50'
default.loss = 'arcface'

default.models_root = './models'
default.pretrained = ''
default.pretrained_epoch = 0
default.ckpt = 2    # 0: discard saving. 1: save when necessary. 2: always save
default.verbose = 2000
default.num_workers = 2
default.lr = 0.1
default.lr_steps = '100000,160000,200000'
default.end_epoch = 20
default.frequent = 20
default.per_batch_size = 128
default.kvstore = 'device'  # device or local
default.opt = 'sgd'

default.auxloss = 'center'
# sequence loss 
default.lsr = False
default.seq_loss_factor = 1.0
# auxiliary loss
default.aux_loss_factor = 1.0
default.memonger = 0

# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r152 = edict()
network.r152.net_name = 'fresnet'
network.r152.num_layers = 152

network.r200 = edict()
network.r200.net_name = 'fresnet'
network.r200.num_layers = 200

network.r50sa = edict()
network.r50sa.net_name = 'resnest'
network.r50sa.emb_size = 512
network.r50sa.num_layers = 50
network.r50sa.bottleneckStr = '2s1x64d'
network.r50sa.deepStem = False
network.r50sa.avgDown = True
network.r50sa.stemWidth = 64
network.r50sa.avd = True
network.r50sa.avdFirst = False
network.r50sa.useSplat = True
network.r50sa.dropblockProb = 0.1
network.r50sa.finalDrop = 0.0
network.r50sa.net_act = 'relu'

network.r101sa = edict()
network.r101sa.net_name = 'resnest'
network.r101sa.emb_size = 512
network.r101sa.num_layers = 101
network.r101sa.bottleneckStr = '2s1x64d'
network.r101sa.deepStem = False
network.r101sa.avgDown = True
network.r101sa.stemWidth = 64
network.r101sa.avd = True
network.r101sa.avdFirst = False
network.r101sa.useSplat = True
network.r101sa.dropblockProb = 0.1
network.r101sa.finalDrop = 0.0
network.r101sa.net_act = 'relu'

network.r200sa = edict()
network.r200sa.net_name = 'resnest'
network.r200sa.emb_size = 512
network.r200sa.num_layers = 200
network.r200sa.bottleneckStr = '2s1x64d'
network.r200sa.deepStem = False
network.r200sa.avgDown = True
network.r200sa.stemWidth = 64
network.r200sa.avd = True
network.r200sa.avdFirst = False
network.r200sa.useSplat = True
network.r200sa.dropblockProb = 0.1
network.r200sa.finalDrop = 0.2
network.r200sa.net_act = 'relu'

network.r269sa = edict()
network.r269sa.net_name = 'resnest'
network.r269sa.emb_size = 512
network.r269sa.num_layers = 269
network.r269sa.bottleneckStr = '2s1x64d'
network.r269sa.deepStem = False
network.r269sa.avgDown = True
network.r269sa.stemWidth = 64
network.r269sa.avd = True
network.r269sa.avdFirst = False
network.r269sa.useSplat = True
network.r269sa.dropblockProb = 0.1
network.r269sa.finalDrop = 0.2
network.r269sa.net_act = 'relu'

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = './dataset/faces_emore'
dataset.emore.num_classes = 85742
dataset.emore.num_training_samples = 5822653
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['cfp_fp','lfw']
#dataset.emore.val_targets = []

dataset.fairface = edict()
dataset.fairface.dataset = 'fairface'
dataset.fairface.dataset_path = './dataset/fairface'
dataset.fairface.num_classes = 4297
dataset.fairface.num_training_samples = 100186
dataset.fairface.image_shape = (112,112,3)
dataset.fairface.val_targets = ['cfp_fp','lfw']
#dataset.fairface.val_targets = []


#dataset.merge_fairface.val_targets = []

dataset.emore_glint = edict()
dataset.emore_glint.dataset = 'emore_glint'
dataset.emore_glint.dataset_path = '/home/ubun/shengyao/data/emore_glint'
dataset.emore_glint.num_classes = 143474
dataset.emore_glint.num_training_samples = 9999999  # wait to be modified
dataset.emore_glint.image_shape = (112,112,3)
dataset.emore_glint.val_targets = ['lfw', 'surveillance']
#dataset.emore_glint.val_targets = []

loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.marginface = edict()
loss.marginface.loss_name = 'svx_softmax'
loss.marginface.loss_s = 64.0
loss.marginface.loss_m1 = 1.0
loss.marginface.loss_m2 = 0.5
loss.marginface.loss_m3 = 0.0
loss.marginface.loss_nm1 = 1.0
loss.marginface.loss_nm2 = 0.0
loss.marginface.loss_nm3 = 0.0
loss.marginface.mask = 1.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.softmax_circle_loss = edict()
loss.softmax_circle_loss.loss_name = 'softmax_circle_loss'
loss.softmax_circle_loss.loss_gamma = 64.0
loss.softmax_circle_loss.loss_margin = 0.25

loss.arc_circle_loss = edict()
loss.arc_circle_loss.loss_name = 'arc_circle_softmax'
loss.arc_circle_loss.loss_gamma = 64.0
loss.arc_circle_loss.thetaMargin = math.pi * 0.65
#loss.arc_circle_loss.thetaMargin = 2.43113

loss.fix_arc_circle_loss = edict()
loss.fix_arc_circle_loss.loss_name = 'fix_arc_circle_softmax'
loss.fix_arc_circle_loss.loss_gamma = 64.0
loss.fix_arc_circle_loss.thetaMargin = math.pi * 0.65



loss.fix_arc_circle_triplet = edict()
loss.fix_arc_circle_triplet.loss_name = 'fix_arc_circle_triplet'
loss.fix_arc_circle_triplet.loss_gamma = 64.0
loss.fix_arc_circle_triplet.thetaMargin = math.pi * 0.65
loss.fix_arc_circle_triplet.images_per_identity = 5
loss.fix_arc_circle_triplet.triplet_alpha = 0.3
loss.fix_arc_circle_triplet.triplet_bag_size = 7200
loss.fix_arc_circle_triplet.triplet_max_ap = 0.0
loss.fix_arc_circle_triplet.per_batch_size = 60
loss.fix_arc_circle_triplet.lr = 0.05


loss.arc_circle_triplet = edict()
loss.arc_circle_triplet.loss_name = 'arc_circle_triplet'
loss.arc_circle_triplet.loss_gamma = 64.0
loss.arc_circle_triplet.thetaMargin = math.pi * 0.65
loss.arc_circle_triplet.images_per_identity = 5
loss.arc_circle_triplet.triplet_alpha = 0.35
loss.arc_circle_triplet.triplet_bag_size = 7200
loss.arc_circle_triplet.triplet_max_ap = 0.0


loss.final_loss = edict()
loss.final_loss.loss_name = 'final_softmax'


loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 7
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 4200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 60
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35# * math.pi
loss.atriplet.triplet_bag_size = 2100
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

loss.triplet_fusion_arc_circle_loss = edict()
loss.triplet_fusion_arc_circle_loss.loss_name = 'triplet_fusion_arc_circle_softmax'
loss.triplet_fusion_arc_circle_loss.images_per_identity = 5
loss.triplet_fusion_arc_circle_loss.triplet_alpha = 0.3
loss.triplet_fusion_arc_circle_loss.triplet_bag_size = 7200
loss.triplet_fusion_arc_circle_loss.triplet_max_ap = 0.0
loss.triplet_fusion_arc_circle_loss.per_batch_size = 200
loss.triplet_fusion_arc_circle_loss.lr = 0.05
loss.triplet_fusion_arc_circle_loss.loss_gamma = 64.0
loss.triplet_fusion_arc_circle_loss.thetaMargin = math.pi * 0.65
loss.triplet_fusion_arc_circle_loss.loss_margin = 0.25

loss.arc_circle_triplet_fusion_arc_circle_loss = edict()
loss.arc_circle_triplet_fusion_arc_circle_loss.loss_name = 'arc_circle_triplet_fusion_arc_circle_softmax'
loss.arc_circle_triplet_fusion_arc_circle_loss.images_per_identity = 5
loss.arc_circle_triplet_fusion_arc_circle_loss.triplet_alpha = 0.3
loss.arc_circle_triplet_fusion_arc_circle_loss.triplet_bag_size = 7200
loss.arc_circle_triplet_fusion_arc_circle_loss.triplet_max_ap = 0.0
loss.arc_circle_triplet_fusion_arc_circle_loss.per_batch_size = 200
loss.arc_circle_triplet_fusion_arc_circle_loss.lr = 0.05
loss.arc_circle_triplet_fusion_arc_circle_loss.loss_gamma = 64.0
loss.arc_circle_triplet_fusion_arc_circle_loss.thetaMargin = math.pi * 0.65
loss.arc_circle_triplet_fusion_arc_circle_loss.loss_margin = 0.25

# auxiliary loss
auxloss = edict()

auxloss.center = edict()
auxloss.center.auxloss_name = 'center'
auxloss.center.center_alpha = 0.05

auxloss.git = edict()
auxloss.git.auxloss_name = 'git'
auxloss.git.center_alpha = 0.05
auxloss.git.git_alpha = 0.01
auxloss.git.git_beta = 0.1
auxloss.git.git_p = 0.001


def generate_config(_network, _dataset, _loss, _auxloss=None):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    if _auxloss != None:
        for k, v in auxloss[_auxloss].items():
            config[k] = v
            if k in default:
                default[k] = v
          
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.auxloss = _auxloss
