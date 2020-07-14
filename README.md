# Fairface-Recognition-Solution

## Baseline

Thie repo is modified from [insightface](https://github.com/deepinsight/insightface)

### Training Data

All traing face images are aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112:

Please check [Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) for detail information and dataset downloading.


* Please check *data_process/face2rec2.py* on how to build a binary face dataset.

### Train

1. Install `MXNet` with GPU support (Python 3.5).

    ```
    pip install mxnet-cu100

    ```
2. Download the training set [MS1M-Arcface](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0) and place it in *`$Fairface-Recognition-Solution-ROOT/train/datasets/`*. Each training dataset includes at least following 6 files:

    ```
    faces_emore/
        train.idx
        train.rec
        property
        lfw.bin
        cfp_fp.bin
        agedb_30.bin
    ```
The first three files are the training dataset while the last three files are verification sets.

3. Train deep face recognition models.

    Edit config file and set you data path and then run

    ```Shell
    ./train.sh
    ```

4. Multi-step fine-tune the above Softmax model .   
Download the trainging set [fairface](http://chalearnlap.cvc.uab.es/dataset/36/description/) and then  build a binary face dataset from it, then you can run 
    ```Shell
    ./fairface_finetune.sh
    ```
    to get the step 1 finetuned model 
    ```Shell
    ./fairface_step2_finetune.sh
    ```
    to get the step2 finetuned model
    ```Shell
    ./fairface_step3_finetune.sh
    ``` 
    to get the step3 finetuned model
       
    It's a multi step finetuing , we freeze all layers but final fc layer at step1 and then finetune all layers at step2 and then use the most discriminated protected data to finetune the model in step3

5. Hard-Sample finetune  
    After we get the finetuned model, we finetune another hard-sample model from the pretrained model (not finetuned) using hard-samples. Hard-samples means samples whose prediction argmax is different from the annotation, and we take all samples from the sub_id of hard-sampels.
       
    Training scripts are same with section4 but we only need 2 steps at this section.
    

### Test



1. Download the pretrained model from [model-zoo](https://1drv.ms/u/s!AoNuuwAvxk2VgztVljlkgMhub_Uy?e=0eHl7D) and test dataset from [fairface](http://chalearnlap.cvc.uab.es/dataset/36/description/) , put the model in *`$Fairface-Recognition-Solution-ROOT/test/final_eval_models`* and data in *`$Fairface-Recognition-Solution-ROOT/test/TestData/tmp_data`* and run 
```Shell
./do.sh
```
