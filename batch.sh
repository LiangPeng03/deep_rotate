#!/bin/bash

gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id

# 定义要测试的模型列表
models=("meta-llama/Meta-Llama-3-8B" "meta-llama/Llama-2-7b-hf")

rotate=(False)

# 双层循环：遍历所有模型和比特数组合
for model in "${models[@]}"; do
    for ro in "${rotate[@]}"; do
        echo "========================================"
        echo "Running: Model=$model, rotate=$ro"
        echo "========================================"
        
        python main.py --model $model \
         --w_bits 3 \
         --w_groupsize 256 \
         --cal_dataset c4 \
         --a_bits 16 \
         --v_bits 16 \
         --k_bits 16 \
         --w_asym \
         --w_clip \
         --act_order \
         --asym_calibrate \
         --enable_aq_calibration \
         --rotate \
         --bsz 1
         
        echo "Finished: Model=$model, rotate=$ro"
    done
done