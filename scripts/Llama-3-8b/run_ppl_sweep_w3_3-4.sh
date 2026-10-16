#!/bin/bash

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate omni

# Set GPU visibility
export CUDA_VISIBLE_DEVICES="2"

# Model configuration
# name="Llama-2-7b-hf"
# name="Llama-3.2-1B-Instruct"
# name="Llama-3.2-3B"
name="Llama-3-8B-w3a16-g128-noalter-3-4"
# name="Llama-2-13b-hf"

# Output directory (change this for different experiments)
output_dir="./log/ppl_sweep/${name}"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Output files (fixed names under the directory)
output_file="$output_dir/ppl_sweep_results.txt"
log_file="$output_dir/ppl_sweep_full_log.txt"

# Initialize log file with header
echo "==========================================" > $log_file
echo "PPL Sweep Started: $(date)" >> $log_file
echo "Model: $name" >> $log_file
echo "==========================================" >> $log_file
echo "" >> $log_file

# Redirect all output to both terminal and log file
exec > >(tee -a "$log_file") 2>&1

# Initialize CSV results
echo "target_bit_eval,actual_bits,ppl" > $output_file

# Sweep target_bit_eval in steps of 0.1
for target_bit in $(seq 3.0 0.1 4.0); do
    echo "=========================================="
    echo "Running evaluation with target_bit_eval = $target_bit"
    echo "=========================================="

    args=(
        --model /mnt/fast7T/models/Llama-3-8B
        # --resume ./log/Llama-3.2-3B
        --resume ./log/Llama-3-8B-w3a16-g128-noalter/Llama-3-8B-noalter
        # --resume ./log/Llama-2-13b-hf
        --epochs 0

        --wbits 2   # MSB bit
        --abits 16

        --batch_size 1
        --lwc
        --group_size 128

        --slice_bits_list 2 2 2 2
        --target_bit_eval $target_bit

        --eval_ppl
    )

    # Run evaluation (output automatically logged)
    python "main_bitmoe.py" "${args[@]}"

    # Extract PPL from recent log output (assuming wikitext2 dataset)
    # Adjust the grep pattern based on your actual dataset name
    ppl=$(tail -100 "$log_file" | grep -oP "wikitext2 : \K[0-9.]+|wikitext-2 : \K[0-9.]+|wikitext : \K[0-9.]+" | tail -1)

    # Extract actual effective bits from log output
    actual_bits=$(tail -100 "$log_file" | grep -oP "Average Effective Bits: \K[0-9.]+" | tail -1)

    if [ -n "$ppl" ] && [ -n "$actual_bits" ]; then
        echo "$target_bit,$actual_bits,$ppl" >> $output_file
        echo "Recorded: target_bit_eval=$target_bit, actual_bits=$actual_bits, PPL=$ppl"
    elif [ -n "$ppl" ]; then
        echo "$target_bit,N/A,$ppl" >> $output_file
        echo "Recorded: target_bit_eval=$target_bit, PPL=$ppl (actual_bits not found)"
    else
        echo "Warning: Could not extract PPL for target_bit_eval=$target_bit"
    fi

    echo ""
done

echo "=========================================="
echo "Sweep complete! Results saved to:"
echo "  - CSV: $output_file"
echo "  - Full log: $log_file"
echo "=========================================="
