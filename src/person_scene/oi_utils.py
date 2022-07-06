import pickle
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw

def oi_maskrect(filepath, annotations, split_name):
    data = pickle.load(open('data/openimages/openimages_{}.pkl'.format(split_name), 'rb'))
    df = pd.read_csv(annotations)

    for i in tqdm(data):
        img = cv2.imread(i)
        h, w, _ = img.shape
        imgId = i.split('/')[-1].split('.')[0]
        bb = df[df['ImageID'] == imgId].iloc[:,4:8]
        xmin, xmax, ymin, ymax = bb.values[0]
        start, end = (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h))
        final_image = cv2.rectangle(img, end, start, (0, 0, 0), -1)
        new_name = filepath.format(split_name, i.split('/')[-1])
        cv2.imwrite(new_name, final_image)
    return 

def oi_maskrect_nobg(filepath, annotations, split_name):
    df = pd.read_csv(annotations)
    df = df[df['IsOccluded'] == 0]
    df = df[df['IsTruncated'] == 0]
    data = pickle.load(open('data/openimages/openimages_{}.pkl'.format(split_name), 'rb'))

    for i in tqdm(data):
        img = cv2.imread(i)
        h, w, _ = img.shape
        imgId = i.split('/')[-1].split('.')[0]
        bbs = df[df['ImageID'] == imgId].iloc[:,3:7]
        black_img = np.zeros(img.shape, np.uint8)
        for bb in bbs.values:
            xmin, xmax, ymin, ymax = bb
            start_x, start_y, end_x, end_y = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
            bb = img[start_y: end_y, start_x: end_x]
            start, end = (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h))
            final_image = cv2.rectangle(black_img, end, start, (255, 255, 255), -1)
        new_name = filepath.format(split_name, i.split('/')[-1])
        cv2.imwrite(new_name, final_image)

