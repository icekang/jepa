#!/bin/bash

source activate jepa
which python

python -m app.main \
  --fname configs/pretrain/vitl16_oct_450.yaml \
  --devices cuda:0 cuda:1