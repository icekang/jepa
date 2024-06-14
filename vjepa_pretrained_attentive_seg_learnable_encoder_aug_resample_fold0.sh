#!/bin/bash

source activate jepa
which python

export OPENBLAS_NUM_THREADS=1
python -u -m evals.main --fname 'configs/evals/segmentation-attentive-aug_resample-fold0 copy.yaml' --devices cuda:0