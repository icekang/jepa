#!/bin/bash

source activate jepa
which python

python -m evals.main --fname configs/evals/segmentation-attentive-fold1.yaml --devices cuda:0