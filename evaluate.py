import pickle, time, argparse
from os import path, mkdir
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nclasses', type=int, default=1)
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--labels_test', type=str, default=None)
    parser.add_argument('--batchsize', type=int, default=170)
    parser.add_argument('--hs', type=int, default=2048)
    parser.add_argument('--device', default=torch.device('cuda'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--res', default=int)
    arg = vars(parser.parse_args())
    
    print('\n', arg, '\n')

    # Load utility files
    humanlabels_to_onehot = pickle.load(open('/n/fs/context-scr/data/COCOStuff/humanlabels_to_onehot.pkl', 'rb'))
    onehot_to_humanlabels = dict((y,x) for x,y in humanlabels_to_onehot.items())
    labels_val_path = '/n/fs/context-scr/data/COCOStuff/labels_val.pkl'
    labels_val = pickle.load(open(labels_val_path, 'rb'))
    gender_labels_test = pickle.load(open(arg['labels_test'], 'rb'))
    
    # Create dataloader
    testset = create_dataset(arg['labels_test'], B=arg['batchsize'], train=False, res=arg['res'])

    # Load model
    classifier = multilabel_classifier(arg['device'], arg['dtype'], nclasses=arg['nclasses'], modelpath=arg['modelpath'], hidden_size=arg['hs'])
    
    # Do inference with the model
    labels_list, scores_list, test_loss_list = classifier.test(testset)

    # Calculate and record mAP
    APs = []
    if arg['nclasses'] == 1:
        APs.append(average_precision_score(labels_list, scores_list))
    else:
        for k in range(arg['nclasses']):
            APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
    mAP = np.nanmean(APs)
    print('mAP (all): {:.2f}'.format(mAP*100.))

    # Given model’s output for each image, get a list of all COCO image ids classified as F/M
    # get image idx that correspond to the gender_label_test.pkl

    F_imgidx = np.where(scores_list>0.5)[0]
    M_imgidx = np.where(scores_list<0.5)[0]
    print('F_imgidx: ', F_imgidx.shape)
    print('M_imgidx: ', M_imgidx.shape)
    
    # use F_imgidx to grab the COCO image ids
    F_coco_img_ids, M_coco_img_ids = [], []
    F_objects, M_objects = [], []
    
    for f_idx in F_imgidx:
        path = list(gender_labels_test.keys())[f_idx]
        img_id = str(path[-12:])
        F_objects.append(labels_val[path])
        F_coco_img_ids.append(path)

    for m_idx in M_imgidx:
        path = list(gender_labels_test.keys())[m_idx]
        img_id = str(path[-12:])
        M_coco_img_ids.append(path)
        M_objects.append(labels_val[path])

    # save F_objects and M_objects
    with open('data/M_objects.pkl', 'wb') as f:
        pickle.dump(M_objects, f)
    
    with open('data/F_objects.pkl', 'wb') as f:
        pickle.dump(F_objects, f)

    with open('data/M_objects_imgid.pkl', 'wb') as f:
        pickle.dump(M_coco_img_ids, f)

    with open('data/F_objects_imgid.pkl', 'wb') as f:
        pickle.dump(F_coco_img_ids, f)
    

if __name__ == "__main__":
    main()
