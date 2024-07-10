#!/bin/bash

source activate jepa
which python

python -u -m evals.main --fname configs/evals/vitl16_cf_pres_16x2x3_learnable_from_segmentation_fold2_high_batch.yaml --devices cuda:1