import matplotlib
import pickle
import pylab
import sys
import os
import json

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pycocotools.coco import COCO
from matplotlib.pyplot import figure as Figure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from skimage.draw import polygon
from tqdm import tqdm

def get_mask_polygons(anns, catIds=[]):
    polygons = []
    for ann in anns:
        if ann['category_id'] in catIds:
            # polygon
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((len(seg)//2, 2))
                polygons.append(Polygon(poly, True,alpha=0.4))
    return polygons    

def mask_objs(imgId, catId, catNm):
    fig = Figure()
    ax = fig.gca()

    I=mpimg.imread('/scratch/network/nmeister/COCO/val2014/COCO_val2014_{}.jpg'.format(imgId))

    # load and display instance annotations
    plt.cla()
    plt.imshow(I); #plt.axis('off')
    annIds = coco.getAnnIds(imgIds=int(imgId), iscrowd=False)
    anns = coco.loadAnns(annIds)
    polygons = get_mask_polygons(anns, [*catId])

    #code to make color average of scene
    colors = []
    if I.ndim == 2:
        I = np.tile(I[..., np.newaxis], (1, 1, 3))
    for channel in range(3):
        colors.append(np.mean(I[:,:,channel])/255.)
    p = PatchCollection(polygons, facecolors=tuple(colors), edgecolors=tuple(colors), linewidths=10)

    ax = plt.gca()
    ax.add_collection(p)
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)

    save_dir = '/scratch/network/nmeister/COCO/{}/'.format(catNm)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig(save_dir + 'COCO_val2014_{}.jpg'.format(imgId), bbox_inches='tight', pad_inches=-0.05)
    plt.close()

def create_pickle(object, data, split):
    output = {}
    # filepath = '/scratch/network/nmeister/COCO/{}/'.format(object)    
    filepath = '/scratch/network/nmeister/COCO/val2014/'

    directory = os.listdir(filepath)
    for file in data: 
        img_name = file.split('/')[-1]
        assert img_name in directory 
        file_name = filepath + img_name
        output[file_name] = data[file]
    pickle.dump(output, open('data/{}_{}.pkl'.format(object, split), 'wb'))

if __name__ == '__main__':
    imgs = pickle.load(open('/scratch/network/nmeister/COCO/data/gender_labels_{}.pkl'.format('test'), 'rb')) 
    create_pickle('baseline', imgs, 'val')
    # coco = COCO('/scratch/network/nmeister/COCO/instances_val2014.json') # add file 
    # splits = ['train', 'val', 'test']
    # objects = ['all']
    # for object in objects:
    #     for split in splits: 
    #         imgs = pickle.load(open('data/gender_labels_{}.pkl'.format(split), 'rb')) 
    #         create_pickle(object, imgs, split)

    # # cats = [(40, 'baseball-glove'), (41, 'skateboard'), (31, 'handbag')]
    # cats = [(np.arange(2, 82), 'all')]
    # for split in splits:
    #     imgs = pickle.load(open('/scratch/network/nmeister/COCO/data/gender_labels_{}.pkl'.format(split), 'rb')) 
    #     for cat in cats: 
    #         catId, catNm = cat
    #         for img in tqdm(imgs): 
    #             img_id = img.split('_')[-1].split('.')[0]
    #             mask_objs(img_id, catId, catNm)  
