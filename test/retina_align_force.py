from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import numpy as np
import random
from time import sleep
from detect_face import RetinaFace
from skimage import transform as trans
import cv2
import pdb
from tqdm import tqdm

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def main(args):
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = os.path.expanduser(args.input_dir)
    multiFacesRF = open(os.path.join(output_dir, 'multi_record.txt'), 'w')
    forceFacesRF = open(os.path.join(output_dir, 'force_record.txt'), 'w')
    
    listFile = open(args.list_file, 'r')
    imgNames = [i.strip() for i in listFile.readlines()]
        
    print('Creating networks and loading parameters')    
    detector = RetinaFace('R50', 0, 0, 'net3')
    
    # configuration for alignment
    threshold = 0.8  # retinaface threshold
    #image_size = [112,96]
    image_size = [112,112]
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041] ], dtype=np.float32 )

    if image_size[1]==112:
        src[:,0] += 8.0
     
    for imgName in tqdm(imgNames):
        imgName = imgName+'.jpg'
        srcImgPath = os.path.join(data_dir, imgName)
        outImgPath = os.path.join(output_dir, imgName)        
         
        try:
            img = cv2.imread(srcImgPath)
            if img is None:
                print("Path is error! ", srcImgPath)
                continue
        except :
            print("Something is error! ", srcImgPath)
        else:                
            
            # fixed to 640*640 by padding
            maxSize = max(img.shape[0], img.shape[1])
            padTop = 0
            padBottom = 0
            padLeft = 0
            padRight = 0
            if img.shape[0] < maxSize:
                rowDiff = maxSize - img.shape[0]
                padTop = rowDiff // 2
                padBottom = rowDiff - padTop
            if img.shape[1] < maxSize:
                colDiff = maxSize - img.shape[1]
                padLeft = colDiff // 2
                padRight = colDiff - padLeft
            
            img = cv2.copyMakeBorder(img,padTop,padBottom,padLeft,padRight,cv2.BORDER_CONSTANT,value=[0,0,0])
            
            fixedSizes = [80,160,320,480,640,800,960]
            aligned_imgs = []
            for fixedSize in fixedSizes:
                scale = float(fixedSize) / float(maxSize)
                bounding_boxes, points = detector.detect(img, threshold, scales=[scale])
                nrof_faces = bounding_boxes.shape[0]
                det = bounding_boxes[:,0:4]
                scores = bounding_boxes[:,4]
                
                img_size = np.asarray(img.shape)[0:2]
                #print(c)
                if (nrof_faces>0): 
                    if nrof_faces > 1:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        
                        bb = np.squeeze(det[index])
                        
                        bb[0] = max(0,bb[0])
                        bb[1] = max(0,bb[1])
                        bb[2] = min(bb[2],img_size[1])
                        bb[3] = min(bb[3],img_size[0])
                        
                        if ((bb[0] >= img_size[1]) or (bb[1] >= img_size[0]) or (bb[2] > img_size[1]) or (bb[3] > img_size[0])):
                            print('Unable to align "%s", bbox error' % image_path)
                            continue
                        
                        h = bb[3]-bb[1]
                        w = bb[2]-bb[0]
                        x = bb[0]
                        y = bb[1] 
                        _w = int((float(h)/image_size[0])*image_size[1] )
                        x += (w-_w)//2
                        #x = min( max(0,x), img.shape[1] )
                        x = max(0,x)
                        xw = x+_w
                        xw = min(xw, img.shape[1])
                        roi = np.array( (x, y, xw, y+h), dtype=np.int32)
                        
                        faceImg = img[roi[1]:roi[3],roi[0]:roi[2],:]
                        dst = points[index, :]
                        tform = trans.SimilarityTransform()
                        tform.estimate(dst, src)
                        M = tform.params[0:2,:]
                        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
                        #M = tform.params
                        #warped = cv2.warpPerspective(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
                        if (warped is None) or (np.sum(warped) == 0):
                            warped = faceImg
                            warped = cv2.resize(warped, (image_size[1], image_size[0]))
                        
                        aligned_imgs.append(warped)
                        multiFacesRF.write(imgName.split('.')[0]+'\n')
                        break
                    else:
                        bb = np.squeeze(det[0])
                        
                        bb[0] = max(0,bb[0])
                        bb[1] = max(0,bb[1])
                        bb[2] = min(bb[2],img_size[1])
                        bb[3] = min(bb[3],img_size[0])
                        
                        if ((bb[0] >= img_size[1]) or (bb[1] >= img_size[0]) or (bb[2] > img_size[1]) or (bb[3] > img_size[0])):
                            continue
                        
                        h = bb[3]-bb[1]
                        w = bb[2]-bb[0]
                        x = bb[0]
                        y = bb[1] 
                        _w = int((float(h)/image_size[0])*image_size[1] )
                        x += (w-_w)//2
                        #x = min( max(0,x), img.shape[1] )
                        x = max(0,x)
                        xw = x+_w
                        xw = min(xw, img.shape[1])
                        roi = np.array( (x, y, xw, y+h), dtype=np.int32)
                        
                        faceImg = img[roi[1]:roi[3],roi[0]:roi[2],:]
                        dst = points[0, :]
                        tform = trans.SimilarityTransform()
                        tform.estimate(dst, src)
                        M = tform.params[0:2,:]
                        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
                        #M = tform.params
                        #warped = cv2.warpPerspective(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
                        if (warped is None) or (np.sum(warped) == 0):
                            warped = faceImg
                            warped = cv2.resize(warped, (image_size[1], image_size[0]))
                        
                        aligned_imgs.append(warped)                          
                        break
        
            if len(aligned_imgs) > 0 :
                cv2.imwrite(outImgPath, aligned_imgs[0])
            else:                
                forceFacesRF.write(imgName.split('.')[0]+'\n')
                    
    multiFacesRF.close()            
    forceFacesRF.close()
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.')  
    parser.add_argument('--list-file', type=str, help='Directory with unaligned images.')    
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

