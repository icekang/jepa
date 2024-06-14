#!/bin/bash

source activate jepa
which python

python -m evals.main --fname configs/evals/segmentation-linear-probe-aug-resample.yaml --devices cuda:0