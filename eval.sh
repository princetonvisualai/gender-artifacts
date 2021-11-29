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
RESOLUTION=224

python evaluate.py --nclasses 1 --modelpath models/$RESOLUTION/lr_0.0001_wd_0.0/model_199.pth --labels_test data/gender_labels_test.pkl --res $RESOLUTION



#OUTDIR='ensemble/CAMS/standard/skateboard/exclusive'
#MODELPATH='/n/fs/context-scr/models/COCOStuff/stage1/standard_80split/model_99.pth'
#python get_cams.py --modelpath $MODELPATH --img_ids ${IMG_IDS[*]} --outdir $OUTDIR
