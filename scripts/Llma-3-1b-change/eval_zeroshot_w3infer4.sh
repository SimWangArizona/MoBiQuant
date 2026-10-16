#!/bin/bash

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate omni

# Set GPU visibility 
export CUDA_VISIBLE_DEVICES="4"

name="Llama-3-1B-w3a16-g128-noalter"
calib_data="wikitext2"
# calib_data="c4_new"
# calib_data="ptb_new"
# calib_data="mix"

# % Evaluation
args=(
    --model /mnt/fast7T/models/Llama-3.2-1B
    --output_dir ./log/zeroshot/${name}
    --resume ./log/Llama-3-1B-w3a16-g128-noalter/Llama-3.2-1B-noalt
    --calib_dataset ${calib_data}

    --wbits 2   # MSB bit
    --abits 16

    --epochs 0
    --lwc
    --group_size 128
    --slice_bits_list 2 2 2 2

    # --slice_bits_list 2 2 2 2
    # --target_activation_ratio_eval 0.33
    --target_bit_eval 4
    --eval_ppl
    # --tasks arc_easy,arc_challenge,boolq,hellaswag,winogrande,piqa,openbookqa,lambada_openai
    --tasks arc_easy,arc_challenge,boolq,hellaswag,winogrande,piqa
    # --tasks arc_challenge

)

python "main_bitmoe.py" "${args[@]}"