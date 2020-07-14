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
    facesInfoRF = open(os.path.join(output_dir, 'info_record.txt'), 'w')
    
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
    
    # alignment process
    nrof_images_total = 0
    nrof = np.zeros( (5,), dtype=np.int32)  # normal, multi, multi_single, force, none
        
    labelDir = data_dir
    target_dir = output_dir
    
    for item in tqdm(os.listdir(labelDir)):
        if not item.endswith('.jpg'):
            continue
        # milestone
        if nrof_images_total%100==0:
            print("Processing %d, (%s)" % (nrof_images_total, nrof))
        nrof_images_total += 1
        #if nrof_images_total<950000:
        #  continue
        
        image_path = os.path.join(labelDir, item)            
        try:
            img = cv2.imread(image_path)
            if img is None:
                print("Path is error! ", image_path)
                continue
        except :
            print("Something is error! ", image_path)
        else:                
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
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
            
            fixedSize = 640
            scale = float(fixedSize) / float(maxSize)
            if scale > 1.0:
                scale = 1.0
            bounding_boxes, points = detector.detect(img, threshold, scales=[scale])
            nrof_faces = bounding_boxes.shape[0]
            det = bounding_boxes[:,0:4]
            scores = bounding_boxes[:,4]
            aligned_imgs = []
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
                    nrof[2]+=1
                    multiFacesRF.write(item.split('.')[0]+'\n')
                    facesInfoRF.write("%s %d\n" % (item.split('.')[0], min(faceImg.shape[0], faceImg.shape[1])))
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
                    nrof[0]+=1
                    facesInfoRF.write("%s %d\n" % (item.split('.')[0], min(faceImg.shape[0], faceImg.shape[1])))
                for i, warped in enumerate(aligned_imgs):
                    #target_file = os.path.join(target_dir, c+str(i)+'.png')
                    target_file = os.path.join(target_dir, item)
                    cv2.imwrite(target_file, warped)
            elif args.detect_force:
                roi = np.zeros( (4,), dtype=np.int32)
                roi[0] = int(img.shape[1]*0.06)
                roi[1] = int(img.shape[0]*0.06)
                roi[2] = img.shape[1]-roi[0]
                roi[3] = img.shape[0]-roi[1]
                warped = img[roi[1]:roi[3],roi[0]:roi[2],:]
                warped = cv2.resize(warped, (image_size[1], image_size[0]))
                #target_file = os.path.join(target_dir, c+'.png')
                target_file = os.path.join(target_dir, item)
                cv2.imwrite(target_file, warped)
                nrof[3]+=1
                forceFacesRF.write(item.split('.')[0]+'\n')
            else:
                print('Unable to detect "%s", face detection error' % image_path)
                nrof[4]+=1
                continue
    forceFacesRF.close()
    multiFacesRF.close()
    facesInfoRF.close()
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', type=str, help='Directory with unaligned images.')    
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    #parser.add_argument('--detect_multiple_faces', action='store_true',
    #                    help='Detect and align multiple faces per image.')
    parser.add_argument('--detect_force', action='store_true',
                        help='Detect and align faces per image forcefully.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

