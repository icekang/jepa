#!/bin/bash

source activate jepa
which python

python -m app.main \
  --fname configs/pretrain/vitl16_oct_tune.yaml \
  --devices cuda:0