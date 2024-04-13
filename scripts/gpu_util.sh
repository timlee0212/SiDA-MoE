#!/bin/bash

for model in switch-base-256 switch-base-128 switch-base-64 switch-base-8
do
    for baseline in 1 2 3
    do
        echo ">>>>" $model "---" $baseline
        python latency.py --baseline $baseline --dataset sst2 --model $model --batch_size 128
    done
done