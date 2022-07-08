import cv2
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw
from IPython.display import display
from tqdm import tqdm
from pycocotools.coco import COCO 
import numpy as np


def get_mask_polygons(anns):
    polygons = []
    for ann in anns:
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((len(seg)//2, 2))
            poly = poly.astype(np.int32)
            polygons.append(poly)
    return polygons

def get_avg_color(image):
    avg_color = []
    for channel in range(3):
        mean_color = np.mean(image[:,:,channel]).astype(np.uint8)
        avg_color.append(mean_color)
        return avg_color

def coco_full_nobg(filepath, annotations, split_name):
    print("Full NoBg")
    files = pickle.load(open('data/mals/{}.pkl'.format(split_name), 'rb'))
    coco = COCO(annotations)

    for filename in tqdm(files):
        image = cv2.imread(filename)
        ann_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), iscrowd=False)
        person_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), catIds=[1], iscrowd=False)
        object_ids = list(set(ann_ids) ^ set(person_ids))
        anns = coco.loadAnns(person_ids)
        object_anns = coco.loadAnns(object_ids)
        object_mask = np.zeros(image.shape, dtype=np.uint8)
        mask = np.zeros(image.shape, dtype=np.uint8)

        black_image = np.zeros(image.shape, np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count

        colors = []
        avg_image = np.zeros(image.shape, np.uint)
        dim1, dim2 = image.shape[0], image.shape[1]
        for channel in range(3):
            mean_color = np.mean(image[:,:,channel]).astype(np.uint8)
            layer = np.full((dim1, dim2), mean_color)
            colors.append(layer)
        avg_image = np.stack(colors)
        avg_image = np.moveaxis(avg_image, 0, -1)


        cv2.fillPoly(object_mask, get_mask_polygons(object_anns), ignore_mask_color)
        cv2.fillPoly(mask, get_mask_polygons(anns), ignore_mask_color)
        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
        final_image = cv2.bitwise_and(image, mask) + cv2.bitwise_and(black_image, mask_inverse)

        img_name = filename.split('/')[-1]
        if not os.path.exists(filepath.format(split_name, '')):
            os.mkdir(filepath.format(split_name, ''))
        new_name = filepath.format(split_name, img_name)
        cv2.imwrite(new_name, final_image)

def coco_masksegm(filepath, annotations, split_name): 
    print("MaskSegm")
    files = pickle.load(open('data/mals/{}.pkl'.format(split_name), 'rb'))
    coco = COCO(annotations)

    for filename in tqdm(files):
        image = cv2.imread(filename)
        ann_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), iscrowd=False)
        person_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), catIds=[1], iscrowd=False)
        object_ids = list(set(ann_ids) ^ set(person_ids))
        anns = coco.loadAnns(person_ids)

        mask = np.zeros(image.shape, dtype=np.uint8)
        black_image = np.zeros(image.shape, np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count

        cv2.fillPoly(mask, get_mask_polygons(anns), ignore_mask_color)
        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
        final_image = cv2.bitwise_and(black_image, mask) + cv2.bitwise_and(image, mask_inverse)

        img_name = filename.split('/')[-1]
        new_name = filepath.format(split_name, img_name)
        cv2.imwrite(new_name, final_image)

def coco_maskrect(filepath, annotations, split_name):
    print("MaskRect")
    files = pickle.load(open('data/mals/{}.pkl'.format(split_name), 'rb'))
    coco = COCO(annotations)

    for filename in tqdm(files):
        image = cv2.imread(filename)
        ann_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), iscrowd=False)
        person_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), catIds=[1], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            x, y, width, height = ann['bbox']
            start, end = (int(x), int(y)), (int(x + width), int(y + height))
            final_image = cv2.rectangle(image, end, start, (0, 0, 0), -1)
        
        img_name = filename.split('/')[-1]
        new_name = filepath.format(split_name, img_name)
        
        cv2.imwrite(new_name, final_image)


def coco_maskrect_nobg(filepath, annotations, split_name):
    print("MaskRect NoBg")
    coco = COCO(annotations)
    files = pickle.load(open('data/mals/{}.pkl'.format(split_name), 'rb'))

    for filename in tqdm(files):
        image = cv2.imread(filename)
        ann_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), iscrowd=False)
        person_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), catIds=[1], iscrowd=False)
        anns = coco.loadAnns(person_ids)
        black_image = np.zeros(image.shape, np.uint8)
        white_image = np.full(image.shape, 255, np.uint8)
        for ann in anns:
            x, y, width, height = ann['bbox']
            x0, y0, x1, y1 = int(x), int(y), int(x + width), int(y + height)
            black_image[y0:y1, x0:x1] = white_image[y0:y1, x0:x1]
        img_name = filename.split('/')[-1]
        new_name = filepath.format(split_name, img_name)
        cv2.imwrite(new_name, black_image)

def coco_masksegm_nobg(filepath, annotations, split_name):
    print("MaskSegm NoBg")
    files = pickle.load(open('data/mals/{}.pkl'.format(split_name), 'rb'))
    coco = COCO(annotations)

    for filename in tqdm(files):
        image = cv2.imread(filename)
        ann_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), iscrowd=False)
        person_ids = coco.getAnnIds(int(filename.split('_')[-1].split('.')[0]), catIds=[1], iscrowd=False)
        object_ids = list(set(ann_ids) ^ set(person_ids))
        anns = coco.loadAnns(person_ids)
        object_anns = coco.loadAnns(object_ids)
        object_mask = np.zeros(image.shape, dtype=np.uint8)
        mask = np.zeros(image.shape, dtype=np.uint8)

        black_image = np.zeros(image.shape, np.uint8)
        white_image = np.full(image.shape, 255, np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count

        colors = []
        avg_image = np.zeros(image.shape, np.uint)
        dim1, dim2 = image.shape[0], image.shape[1]
        for channel in range(3):
            mean_color = np.mean(image[:,:,channel]).astype(np.uint8)
            layer = np.full((dim1, dim2), mean_color)
            colors.append(layer)
        avg_image = np.stack(colors)
        avg_image = np.moveaxis(avg_image, 0, -1)


        cv2.fillPoly(object_mask, get_mask_polygons(object_anns), ignore_mask_color)
        cv2.fillPoly(mask, get_mask_polygons(anns), ignore_mask_color)
        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask
        final_image = cv2.bitwise_and(black_image, mask) + cv2.bitwise_and(white_image, mask_inverse)

        img_name = filename.split('/')[-1]
        new_name = filepath.format(split_name, img_name)
        cv2.imwrite(new_name, final_image)

