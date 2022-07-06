import pickle, time, argparse, random
from os import path, makedirs
import numpy as np
import torch
import json

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

from classifier import multilabel_classifier
from load_data import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--pretrainedpath', type=str, default=None)
    parser.add_argument('--outdir', type=str, default='save')
    parser.add_argument('--nclasses', type=int, default=1)
    parser.add_argument('--labels_train', type=str, default=None)
    parser.add_argument('--labels_val', type=str, default=None)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--train_batchsize', type=int, default=200)
    parser.add_argument('--val_batchsize', type=int, default=170)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--drop', type=int, default=60)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--hs', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--res', type=int, default=224)
    
    arg = vars(parser.parse_args())
    #json.dump(arg, open('{}/args.json'.format(arg['outdir']), 'w'))
    print('\n', arg, '\n')
    print('\nTraining with {} GPUs'.format(torch.cuda.device_count()))

    # Set random seed
    random.seed(arg['seed'])
    np.random.seed(arg['seed'])
    torch.manual_seed(arg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create output directory
    if not path.isdir(arg['outdir']):
        makedirs(arg['outdir'])

    # Load utility files
    
    # Create data loaders
    trainset = create_dataset(arg['labels_train'], 
                              B=arg['train_batchsize'], train=True, res=arg['res'])
    valset = create_dataset(arg['labels_val'],
                             B=arg['val_batchsize'], train=False, res=arg['res'])

    # Initialize classifier
    classifier = multilabel_classifier(arg['device'], arg['dtype'], nclasses=arg['nclasses'],
                                       modelpath=arg['modelpath'], hidden_size=arg['hs'], learning_rate=arg['lr'], model_file=arg['pretrainedpath'])
    classifier.epoch = 1 # Reset epoch for stage 2 training
    classifier.optimizer = torch.optim.SGD(classifier.model.parameters(), lr=arg['lr'], momentum=arg['momentum'], weight_decay=arg['wd'])

    # Keep track of loss and mAP/recall for best model selection
    loss_epoch_list = []; all_list = []
    min_loss = 10000
    # Start training
    tb = SummaryWriter(log_dir='{}/runs'.format(arg['outdir']))
    start_time = time.time()
    print('\nStarted training at {}\n'.format(start_time))
    for i in range(1, arg['nepoch']+1):

        # Reduce learning rate from 0.1 to 0.01
        train_loss_list = classifier.train(trainset)
        
        # Save the model
        if (i + 1) % 10 == 0:
            classifier.save_model('{}/model_{}.pth'.format(arg['outdir'], i))

        # Do inference with the model
        labels_list, scores_list, val_loss_list, _ = classifier.test(valset)
        
        # Record train/val loss
        tb.add_scalar('Loss/Train', np.mean(train_loss_list), i)
        tb.add_scalar('Loss/Val', np.mean(val_loss_list), i)
        #if np.mean(val_loss_list) <= min_loss:
        #    min_loss = np.mean(val_loss_list)
        #    classifier.save_model('{}/model_best.pth'.format(arg['outdir']))
        loss_epoch_list.append(np.mean(val_loss_list))

        # Calculate and record mAP
        APs = []
        if arg['nclasses'] == 1:
            APs.append(average_precision_score(labels_list, scores_list))
        else:
            for k in range(arg['nclasses']):
                APs.append(average_precision_score(labels_list[:,k], scores_list[:,k]))
        mAP = np.nanmean(APs)
        tb.add_scalar('mAP/all', mAP*100, i)

        all_list.append(mAP*100)

        
        # Print out information
        print('\nEpoch: {}'.format(i))
        print('Loss: train {:.5f}, val {:.5f}'.format(np.mean(train_loss_list), np.mean(val_loss_list)))
        print('Val mAP: all {} {:.5f}'.format(arg['nclasses'], mAP*100))
        print('Time passed so far: {:.2f} minutes\n'.format((time.time()-start_time)/60.))

    # Print best model and close tensorboard logger
    tb.close()
    print('Best model at {} with lowest val loss {}'.format(np.argmin(loss_epoch_list) + 1, np.min(loss_epoch_list)))
    
if __name__ == "__main__":
    main()
