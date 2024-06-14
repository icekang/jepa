#!/bin/bash

source activate jepa
which python

export OPENBLAS_NUM_THREADS=1

python -m app.main \
  --fname configs/pretrain/vitl16_oct.yaml \
  --devices cuda:0