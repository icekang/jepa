#!/bin/bash

source activate jepa
which python

python -u -m evals.main --fname configs/evals/vitl16_cf_pres_tabular_loader_learnable_from_segmentation_fold0.yaml --devices cuda:0