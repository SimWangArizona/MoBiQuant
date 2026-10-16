#!/bin/bash


export CUDA_VISIBLE_DEVICES="5"



name="Llama-2-7b-hf-w4a16-g128-LWC-MoE"

# % Training
args=(
    --model /mnt/fast7T/models/Llama-2-7b-hf
    --output_dir ./log/${name}

    --wbits 2   # MSB bit
    --attnbits 4 # attn block bit
    --abits 16
    # --onlymlp

    --epochs 40
    --batch_size 1
    --lwc
    --lwc_lr 1e-3
    --aug_loss
    --group_size 128
    
    --slice_bits_list 2 2 2 2
    --stage_weight 1.0 1.33
    --bitmoe_lr 8e-5
    --reg_weight 8e-5
    --target_activation_ratio 0.35

    --eval_ppl
)

# % Evaluation
# args=(
#     --model meta-llama/${name}
#     --output_dir ./log/test
#     --resume ./log/test

#     --wbits 2   # MSB bit
#     --abits 16

#     --epochs 0
#     --lwc
#     --group_size 128

    # --target_bit_eval ??
    # --target_activation_ratio_eval ??
# )

python "main_bitmoe.py" "${args[@]}"