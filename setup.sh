#!/bin/bash

# Create a conda environment named moe
conda create -n moe python=3.8

# Activate the environment
source activate moe

# Install packages
pip install torch
pip install transformers
pip install datasets
pip install wandb
pip install torchvision
pip install scipy
pip install scikit-learn
pip install transformers[torch]
pip install seaborn
pip install deepspeed