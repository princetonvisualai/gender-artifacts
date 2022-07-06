#!/bin/sh
#SBATCH -N 1                   # number of nodes requested                                                                                         
#SBATCH -n 1                   # number of tasks requested                                                                                             
#SBATCH --ntasks-per-node 1    # number of tasks per node                                                                                              
#SBATCH --exclude=node718      # exclude the node that often causes errors                                                                             
#SBATCH -A visualai            # specify which group of nodes to use                                                                                   
#SBATCH --mem-per-cpu=4G       # memory per cpu-core (4G default)                                                                                      
#SBATCH --gres=gpu:rtx_3090:1  # number of GPUs requested                                                                                              
#SBATCH -t 60:00:00            # time requested in hour:minute:second                                                                                  


source /n/fs/context-scr/context/bin/activate # for RTX3090              

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.00001 --res 128

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.000001 --res 128

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.00001 --res 64

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.000001 --res 64

python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.00001 --res 32

python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.000001 --res 32

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0.00001 --momentum 0.9 --lr 0.1 --res 128

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0.0001 --momentum 0.9 --lr 0.1 --res 128

#python train.py --nclasses 1 --outdir models/ --labels_train data/gender_labels_train.pkl --labels_val data/gender_labels_val.pkl \
#  --nepoch 200 --wd 0 --momentum 0.9 --lr 0.00001 --res 128  --nepoch 200 --wd 0.001 --momentum 0.9 --lr 0.1 --res 128


#OUTDIR='ensemble/CAMS/standard/skateboard/exclusive'
#MODELPATH='/n/fs/context-scr/models/COCOStuff/stage1/standard_80split/model_99.pth'
#python get_cams.py --modelpath $MODELPATH --img_ids ${IMG_IDS[*]} --outdir $OUTDIR
