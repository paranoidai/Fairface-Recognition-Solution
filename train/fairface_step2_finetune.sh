#!/usr/bin/env bash
#export MXNET_CPU_WORKER_NTHREADS=24
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
export MXNET_SAFE_ACCUMULATION=1
#export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
# arcface LResNet100E-IR
#CUDA_VISIBLE_DEVICES='0' nohup python -u train_seqface.py --network r100 --loss arcface --dataset multidata --auxloss git --per-batch-size 420 > a.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' python -u train_seqface.py --network r100 --loss arcface --dataset multidata --auxloss git --per-batch-size 128
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r100 --loss svxface --dataset emore --per-batch-size 440 > c.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r100 --loss arcface --dataset emore_glint --per-batch-size 420 > a.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r50 --loss svxface --dataset emore_glint --per-batch-size 480 > 128_svxface_emore_glint_20190626.out 2>&1 &
#CUDA_VISIBLE_DEVICES='0' python -u train.py --network r50 --loss svxface --dataset emore_glint --per-batch-size 480
#CUDA_VISIBLE_DEVICES='0' nohup python -u train.py --network r100 --loss svxface --dataset emore_glint --per-batch-size 400 > a.out 2>&1 &

#CUDA_VISIBLE_DEVICES='5' python -u train.py --network r50 --loss marginface --dataset emore --per-batch-size 200
# triplet LResNet100E-IR
#CUDA_VISIBLE_DEVICES='0' python3 -u train.py --network m3 --loss arc_circle_loss --lr 0.1 --dataset emore --per-batch-size 400 --verbose 2000 --pretrained /home/ubuntu/shengyao/eccv_fair_face_recognition/pre_trained_models/model

CUDA_VISIBLE_DEVICES='0' python3 -u train.py --network r100 --loss arc_circle_loss --lr 0.002 --dataset fairface --per-batch-size 400 --pretrained ./models/20200630235820-r100-arc_circle_loss-fariface/model --pretrained-epoch 3 --verbose 800 --lr-steps '6000,12000,18000' 
#CUDA_VISIBLE_DEVICES='0' python3 -u train.py --network r100 --loss atriplet --lr 0.0002 --dataset fairface --per-batch-size 300 --pretrained /home/ubuntu/shengyao/pretrained_models/r100_arc_circle_train_triplet_finetune/model --verbose 200 --lr-steps '3000,6000,10000'

# parall train
#CUDA_VISIBLE_DEVICES='0' python -u train_parall.py --network r100 --loss arcface --dataset emore

