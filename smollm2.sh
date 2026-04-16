#!/bin/bash

gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id

$HOME/.conda/envs/awq/bin/python main.py --model HuggingFaceTB/SmolLM2-135M \
 --w_bits 16 \
 --w_groupsize 256 \
 --cal_dataset c4 \
 --a_bits 16 \
 --v_bits 16 \
 --k_bits 16 \
 --w_asym \
 --asym_calibrate \
 --w_clip \
#  --rotate
