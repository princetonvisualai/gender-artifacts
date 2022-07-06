#!/bin/sh

python train.py \
--nclasses 1 \
--outdir models/ \
--labels_train $1 \
--labels_val $2 \
--nepoch 200 \
--wd 0 \
--momentum 0.9 \
--lr 0.00001
