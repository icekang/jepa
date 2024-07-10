#!/bin/bash

source activate jepa
which python

# export OPENBLAS_NUM_THREADS=1

python -u -m evals.main --fname vitl16_MCCS_MSCT_tabular_loader_learnable_from_segmentation_fold0.yaml --devices cuda:0