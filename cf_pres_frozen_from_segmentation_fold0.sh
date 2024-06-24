#!/bin/bash

source activate jepa
which python

export OPENBLAS_NUM_THREADS=1

python -u -m evals.main --fname configs/evals/vitl16_cf_pres_16x2x3_from_segmentation.yaml --devices cuda:1