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
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--res', default=int)
    parser.add_argument('--outfile', default=str)
    arg = vars(parser.parse_args())
    
    print('\n', arg, '\n')

    # Load utility files
    # labels_val = pickle.load(open(labels_val_path, 'rb'))
    gender_labels_test = pickle.load(open(arg['labels_test'], 'rb'))
    
    # Create dataloader
    testset = create_dataset(arg['labels_test'], B=arg['batchsize'], train=False, res=arg['res'])

    # Load model
    classifier = multilabel_classifier(arg['device'], arg['dtype'], nclasses=arg['nclasses'], modelpath=arg['modelpath'], hidden_size=arg['hs'], model_file=None)
    
    # Do inference with the model
    labels_list, scores_list, test_loss_list, files_list = classifier.test(testset)
    
    output = {'files': files_list, 'labels': labels_list, 'scores': scores_list, 'loss': test_loss_list}
    pickle.dump(output, open(arg['outfile'], 'wb'))

    # Calculate and record mAP
    APs = []
    if arg['nclasses'] == 1:
        APs.append(average_precision_score(labels_list, scores_list))
    else:
        for k in range(arg['nclasses']):
            APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
    mAP = np.nanmean(APs)
    print('mAP (all): {:.2f}'.format(mAP*100.))


if __name__ == "__main__":
    main()
