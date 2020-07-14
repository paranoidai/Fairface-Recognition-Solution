from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datetime import datetime
import os.path
from easydict import EasyDict as edict
import time
import json
import sys
import numpy as np
import importlib
import itertools
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import mxnet as mx
from mxnet import ndarray as nd
import pdb
from tqdm import tqdm

def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_feature(buffer, net, args):
    imageShape = [int(x) for x in args.image_size.split(',')]    
    useFlip = True
    
    if useFlip:
        inputBlob = np.zeros( (len(buffer)*2, 3, imageShape[1], imageShape[2]) )
    else:
        inputBlob = np.zeros( (len(buffer), 3, imageShape[1], imageShape[2]) )
    idx = 0
    for item in buffer:
        # get aligned face
        img = cv2.imread(item)
        img = cv2.resize(img,(112,112))
        # bgr to rgb
        img = img[...,::-1]
        img = np.transpose( img, (2,0,1) )
        attempts = [0,1] if useFlip else [0]
        for flipid in attempts:
            _img = np.copy(img)
            if flipid==1:
                do_flip(_img)
            inputBlob[idx] = _img
            idx+=1
    data = mx.nd.array(inputBlob)
    db = mx.io.DataBatch(data=(data,))
    net.model.forward(db, is_train=False)
    _embedding = net.model.get_outputs()[0].asnumpy()
    embSize = _embedding.shape[1]
    embedding = np.zeros( (len(buffer), embSize), dtype=np.float32 )
    if useFlip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = embedding1+embedding2
    else:
        embedding = _embedding
    embedding = sklearn.preprocessing.normalize(embedding)
    return embedding



def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        #key = os.path.splitext(each)[0] #.jpg
        key = each.split('.')[0] #.jpg0.png or .jpg.png
        fe_dict[key] = features[i]
    return fe_dict

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def load_feature_model(ctx, args):
    imageShape = [int(x) for x in args.image_size.split(',')]
    vec = args.model.split(',')
    assert len(vec)>1
    prefix = vec[0]
    epoch = int(vec[1])
    print('loading', prefix, epoch)
    net = edict()
    net.ctx = ctx
    net.sym, net.argParams, net.auxParams = mx.model.load_checkpoint(prefix, epoch)
    allLayers = net.sym.get_internals()
    net.sym = allLayers['fc1_output']
    #net.sym = allLayers['fusion_fc1_output']
    net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names = None)
    net.model.bind(data_shapes=[('data', (args.batch_size, 3, imageShape[1], imageShape[2]))])
    net.model.set_params(net.argParams, net.auxParams)
    
    return net

def predict_features(imgPaths, net, args, embSize):
    s = time.time()
    i = 0
    fstart = 0
    buffer = []
    dataSize = len(imgPaths)
    # predict face features
    featuresAll = None
    for path in imgPaths:
        if i%1000==0:
            print("processing ",i)
        i+=1
        buffer.append(path)
        if len(buffer)==args.batch_size:
            embedding = get_feature(buffer, net, args)
            buffer = []
            fend = fstart+embedding.shape[0]
            if featuresAll is None:
                featuresAll = np.zeros( (dataSize, embSize), dtype=np.float32 )
            #print('writing', fstart, fend)
            featuresAll[fstart:fend,:] = embedding
            fstart = fend
    if len(buffer)>0:
        embedding = get_feature(buffer, net, args)
        fend = fstart+embedding.shape[0]
        #print('writing', fstart, fend)
        featuresAll[fstart:fend,:] = embedding    
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / dataSize))
    
    return featuresAll

def main(args):
    # print current args
    print(args)
    embSize = 512
    # gpu or cpu contexts
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
    
    # load model
    net = load_feature_model(ctx, args)

    # load face data
    org_nameList = [name for name in os.listdir(args.input_dir)]
    nameList = []
    for item in org_nameList:
      if('jpg' in item or 'png' in item):
        nameList.append(item)
    
    imgPaths = [args.input_dir+'/'+name for name in nameList]
    print('Faces number:', len(imgPaths))
  
    # predict faces features
    featuresAll = predict_features(imgPaths, net, args, embSize)
    # calculate final results
    faceFeatures = get_feature_dict(nameList, featuresAll)
    print('Output number:', len(faceFeatures))
    sampleFile = open(args.output_dir+'/predictions.csv', 'r')
    outFile = open(args.output_dir+'/'+args.file_name, 'w')
    print('Loaded CSV')
    
    lines = sampleFile.readlines()
    pbar = tqdm(total=len(lines))
    idx = 0
    for line in lines:
        idx += 1
        if idx == 1:
            outFile.write(line)
            continue        

        a, b, score = line.split(',')        
        score = '%.4f' % cosin_metric(faceFeatures[a], faceFeatures[b])
        outFile.write(a+','+b+','+score+'\n')
        pbar.update(1)
    
    sampleFile.close()
    outFile.close()
    print("============processed done!=============")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=32)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--input_dir', type=str, help='', default='')
    parser.add_argument('--output_dir', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')
    parser.add_argument('--file_name', type=str, help='', default='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


