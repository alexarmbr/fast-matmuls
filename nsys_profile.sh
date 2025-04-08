#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nsys profile \
    -w true \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s cpu \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cudabacktrace=true \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-set=gh100 \
    -x true \
    -f true \
    -o profiling/kernel1 \
    -e PROFILE_KERNEL=1 \
    python benchmark_kernel.py