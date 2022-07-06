#!/bin/sh

python evaluate.py \
--nclasses 1 \
--modelpath $1 \
--labels_test $2 

