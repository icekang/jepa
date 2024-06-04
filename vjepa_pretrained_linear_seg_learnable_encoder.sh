#!/bin/bash

source activate jepa
which python

python -m evals.main --fname configs/evals/segmentation-linear-probe.yaml --devices cuda:0