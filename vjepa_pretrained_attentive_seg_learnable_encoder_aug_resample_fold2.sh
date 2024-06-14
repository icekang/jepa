#!/bin/bash

source activate jepa
which python
export OPENBLAS_NUM_THREADS=1
python -m evals.main --fname 'configs/evals/segmentation-attentive-aug_resample-fold2 copy.yaml' --devices cuda:0