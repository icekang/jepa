#!/bin/bash

source activate jepa
which python

python -m evals.main --fname configs/evals/segmentation.yaml --devices cuda:0