#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=4
#PJM -L elapse=120:00:00
#PJM -j

module load cuda

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Run Python script
python llama_train.py