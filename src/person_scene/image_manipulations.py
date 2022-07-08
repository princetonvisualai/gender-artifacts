import argparse 
import os 

from oi_utils import *
from coco_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'openimages'])
    parser.add_argument('--type', type=str, 
        choices=['full', 'rect', 'segm'], default='full', help='What manipulation to the person')
    parser.add_argument('--background', action=argparse.BooleanOptionalAction)
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--annotations', type=str)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    args = vars(parser.parse_args())

    filepath = args['filepath']
    split = args['split']
    annotations = args['annotations']
    if not os.path.exists(filepath.format(split, '')):
        os.mkdir(filepath.format(split, ''))

    if args['dataset'] == 'coco':
        if args['background']:
            if args['type'] == 'full': print('Full with background is the original image')
            elif args['type'] == 'rect': coco_maskrect(filepath, annotations, split)
            elif args['type'] == 'segm': coco_masksegm(filepath, annotations, split)
            else: print('Choose a type from rect or segm')
        else:
            if args['type'] == 'full': coco_full_nobg(filepath, annotations, split)
            elif args['type'] == 'rect': coco_maskrect_nobg(filepath, annotations, split)
            elif args['type'] == 'segm': coco_masksegm_nobg(filepath, annotations, split)
            else: print('Choose a type from full or rect or segm')
    elif args['dataset'] == 'openimages':
        if args['background']:
            if args['type'] == 'full': print('Full with background is the original image')
            elif args['type'] == 'rect': oi_maskrect(filepath, annotations, split)
            else: print('Choose type rect')
        else:
            if args['type'] == 'rect': oi_maskrect_nobg(filepath, annotations, split)
            else: print('Choose type rect')
    else:
        print('Dataset must be either coco or openimages')

if __name__ == '__main__':
    main()

