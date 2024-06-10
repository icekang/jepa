#!/bin/bash

source activate jepa
which python
export OPENBLAS_NUM_THREADS=1
python -m evals.main --fname configs/evals/segmentation-attentive-aug_resample-fold1.yaml --devices cuda:1