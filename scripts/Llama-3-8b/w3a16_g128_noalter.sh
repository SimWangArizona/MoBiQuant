#!/bin/bash

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate omni

# Set GPU visibility
export CUDA_VISIBLE_DEVICES="5"

# name="Llama-2-7b-hf"
# name="Llama-3.1-8B"
name="Llama-3-8B-w3a16-g128-noalter"
args=(
    --model /mnt/fast7T/models/Llama-3-8B
    --output_dir ./log/${name}
    ## Eval only (comment out original epoch, and set epoch=0):
    # --resume ./log/Llama-3-1B-w4a16-g128-original
    # --epochs 0

    --wbits 2   # MSB bit

    --abits 16
    # --onlymlp ## if you want to quantize onlymlp

    --epochs 20
    --batch_size 1
    --lwc
    --lwc_lr 1e-3
    --aug_loss
    --group_size 128
    
    --slice_bits_list 2 2 2 2
    --stage_weight 1.0 1.33
    --bitmoe_lr 8e-5
    --reg_weight 8e-5
    ## for static inference, there's no need to set any of the below. 
    ## The router will route to how it was trained by default. 
    --target_activation_ratio 0.15 
    ## for flexible inference:
    # --target_bit_eval 3.0
    # --target_activation_ratio_eval 0.15
    
    # --wandb
    --eval_ppl
)

python "main_bitmoe.py" "${args[@]}"