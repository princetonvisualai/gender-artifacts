import pickle
import os
import cv2 
import PIL
import numpy as np 

from PIL import Image, ImageDraw
from tqdm import tqdm 
from pycocotools.coco import COCO 

SPLIT = 'train'
coco = COCO('/n/fs/visualai-scr/Data/Coco/2014data/annotations/instances_{}2014.json'.format(SPLIT))

def blackout_person(ann, file, imgId):
    image = cv2.imread(file)
    black_image = np.zeros(image.shape, np.uint8)

    mask = np.zeros(image.shape, dtype=np.uint8)
    channel_count = image.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, get_mask_polygons(ann), ignore_mask_color)
    mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
    final_image = cv2.bitwise_and(black_image, mask) + cv2.bitwise_and(image, mask_inverse)
    
    return final_image

def get_mask_polygons(anns):
    polygons = []
    for ann in anns:
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((len(seg)//2, 2))
            poly = poly.astype(np.int32)
            polygons.append(poly)
    return polygons

def blur_person(ann, file, imgId):
    image = cv2.imread(file)
    blurred_image = cv2.GaussianBlur(image,(25, 25), 0)

    mask = np.zeros(image.shape, dtype=np.uint8)
    channel_count = image.shape[2]
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, get_mask_polygons(ann), ignore_mask_color)
    mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
    final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image, mask_inverse)
    
    return final_image

def create_blurred(annotations):
    data = pickle.load(open(annotations, 'rb'))
    labels = {}
    filepath = '/n/fs/nmdz-gender/data/blur/{}/{}'
    blurred_ims = []
    
    for i in tqdm(data):
        imgName = i.split('/')[-1]
        imgId = i.split('_')[-1]
        
        # images without people 
        id_num = int(imgId.split('.')[0])
        filename = filepath.format(SPLIT, imgName)
        if not os.path.exists(filename):
                # save blurred image
            anns = coco.loadAnns(ids=coco.getAnnIds(imgIds=[id_num], catIds=[1], iscrowd=False))
            img = blur_person(anns, i, imgId)
            cv2.imwrite(filename, img)
       	    labels[filename] = data[i]
            blurred_ims.append(filename)
    return blurred_ims

def create_blackout(annotations):
    data = pickle.load(open(annotations, 'rb'))
    labels = {}
    filepath = '/n/fs/nmdz-gender/data/blackout/{}/{}'
    black_ims = []

    for i in tqdm(data):
        imgName = i.split('/')[-1]
        imgId = i.split('_')[-1]

        # images without people
        id_num = int(imgId.split('.')[0])
        filename = filepath.format(SPLIT, imgName)
        if not os.path.exists(filename):
                # save blurred image
            anns = coco.loadAnns(ids=coco.getAnnIds(imgIds=[id_num], catIds=[1], iscrowd=False))
            img = blackout_person(anns, i, imgId)
            cv2.imwrite(filename, img)
            labels[filename] = data[i]
            black_ims.append(filename)
    return black_ims

def main(file):
    imgs = create_blurred(file)
    print('Total Imgs: {}'.format(len(imgs)))
    imgs = create_blackout(file)
    print('Total Imgs: {}'.format(len(imgs)))

if __name__ == "__main__":
    main('mals/train.pkl')
