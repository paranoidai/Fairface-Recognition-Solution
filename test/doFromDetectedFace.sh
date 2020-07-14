#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

SRC_DATA_DIR=../TestData/tmp_data
OUT_FACE_DATA_DIR=../TestData/retinaface
OUT_FACE_DATA_F_DIR=../TestData/retinaface_f
OUT_FACE_DATA_R_DIR=../TestData/retinaface_r

# preprocess source data
## 1. detection with retinaface
#python3 retina_align.py --input-dir ${SRC_DATA_DIR} --output-dir ${OUT_FACE_DATA_DIR} --detect_force

## 2. force detection on part data
#python3 retina_align_force.py --input-dir ${SRC_DATA_DIR} --output-dir ${OUT_FACE_DATA_F_DIR} --list-file ${OUT_FACE_DATA_DIR}/force_record.txt

## 3. rotate detection on part data
#python3 retina_align_rotate_force.py --input-dir ${SRC_DATA_DIR} --output-dir ${OUT_FACE_DATA_R_DIR} --list-file ${OUT_FACE_DATA_F_DIR}/force_record.txt

## merge data
#cp "${OUT_FACE_DATA_R_DIR}/"*.jpg ${OUT_FACE_DATA_DIR}
#cp "${OUT_FACE_DATA_F_DIR}/"*.jpg ${OUT_FACE_DATA_DIR}

#run prediction on the hard sample model
python3 gen_eccv.py --input_dir ${OUT_FACE_DATA_DIR} --output_dir ./shengyao_results --file_name predictions_hard_positive.csv --model ../final_eval_models/hard_sample_train/model,43 --batch_size 10
#run the prediction on the selected attribute finetune model
python3 gen_eccv.py --input_dir ${OUT_FACE_DATA_DIR} --output_dir ./shengyao_results --file_name predictions_dark_female_3_step_finetune.csv --model ../final_eval_models/dark_female_3_steps_finetune/model,1 --batch_size 10

#reranking
python3 dis_reranking.py ./shengyao_results/predictions_dark_female_3_step_finetune.csv
#boundry cut
python3 find_grad_cut.py ./reranked.csv 
#hard sample model result fusion
python3 result_fusion.py ./shengyao_results/predictions_hard_positive.csv grad_cut.csv
