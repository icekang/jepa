#!/bin/bash

source activate jepa
which python

python -m evals.main --fname configs/evals/segmentation-attentive-aug_resample-fold0.yaml --devices cuda:0