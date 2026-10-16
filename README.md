# MoBiQuant

Selected BitMoE training and evaluation code from BitMoEV2 branch `buildq2`, commit `7d493d8ffe7a4367b8f99375a9599547dab5ce99`. Original source files are preserved byte for byte; see `MIGRATION_MANIFEST.json`.

## Architecture

`main_bitmoe.py` + `parser.py` → `models/LMClass.py` and `datautils.py` → `quantize/bitmoequant_llama3.py` → `models/int_llama_layer.py` → `quantize/int_moe.py` and `quantize/quantizer.py`.

With `--slice_bits_list 2 2 2 2` and without `--onlymlp`, all decoder layers use BitMoE for Attention q/k/v/o and MLP gate/up/down projections. Embeddings, norms and the language head are not converted to MoE. Experts represent base quantization plus routed residual weights, rather than native Mixtral experts. Mixtral is unsupported by the active workflow.

## Environment

The source machine's working environment is `omniquant`: Python 3.10.18, PyTorch 2.8.0+cu128, Transformers 4.56.1. `omniquant2` uses Transformers 4.33.0 and fails the active code's `LlamaRotaryEmbedding(config=...)` call. Additional runtime package versions are recorded in `validation/environment.json`.

On the source machine:

```bash
conda activate omniquant
python validation/gpu_smoke.py
```

For a new environment, install a suitable CUDA PyTorch build and the project dependencies from `pyproject.toml`; also install `termcolor`, which the logger imports. Pin Transformers to 4.56.1 for the validated API. The project metadata's broad minimum versions are inherited and are not a reproducible lockfile. AutoGPTQ is optional for this floating-point simulation workflow; its absence prints a message. Real packed quantization has not been validated.

## Train and evaluate

Run from the repository root. Replace model, GPU and output paths as needed. Original model weights and Hugging Face dataset access/cache are required for real model runs.

```bash
CUDA_VISIBLE_DEVICES=0 python main_bitmoe.py \
  --model /mnt/fast7T/models/Llama-3-8B \
  --output_dir ./log/Llama-3-8B-bitmoe \
  --wbits 2 --abits 16 --group_size 128 \
  --slice_bits_list 2 2 2 2 --stage_weight 1.0 1.33 \
  --epochs 20 --batch_size 1 --lwc --lwc_lr 1e-3 \
  --bitmoe_lr 8e-5 --reg_weight 8e-5 \
  --target_activation_ratio 0.15 --eval_ppl
```

Restore trained parameters and evaluate at a target average bit rate:

```bash
CUDA_VISIBLE_DEVICES=0 python main_bitmoe.py \
  --model /mnt/fast7T/models/Llama-3-8B \
  --resume ./log/Llama-3-8B-bitmoe --output_dir ./log/eval \
  --epochs 0 --wbits 2 --abits 16 --group_size 128 \
  --slice_bits_list 2 2 2 2 --target_bit_eval 4 --eval_ppl
```

Directory resume requires `omni_parameters.pth`, `router_parameters.pth` and `router_stats.pth`. These contain calibration/router state, not the base model. Even `--epochs 0` loads calibration data. Use the same model and quantization geometry as training. Add `--tasks arc_easy,arc_challenge,boolq,hellaswag,winogrande,piqa` for downstream evaluation.

## Retained experiment scripts

- `scripts/Llama-3-8b/`: 8B training, restored zero-shot evaluation and bit/PPL sweeps.
- `scripts/Llama-3-3b-change/`: 3B training and evaluation.
- `scripts/Llma-3-1b-change/`: 1B training and evaluation (original directory spelling retained).
- `scripts/Llama-2-7B-change/`: Llama-2 7B training and evaluation.

Only scripts invoking `main_bitmoe.py` are retained. They are historical configurations: inspect hardcoded GPUs, paths and checkpoint names before running. Script names such as w3/w4 describe experiment targets, not necessarily `--wbits`, which is usually 2 here.

## Known implementation limits

- The active layer is `models/int_llama_layer.py`. Alternative whole-MoE/LWC modules are retained for reference; changing their import changes the architecture.
- `--attnbits` does not change the active default Attention implementation. It is used by the inactive LWC variant.
- W&B imports are commented out in the active entry/calibration files. Leave `--wandb` disabled. W&B sweep launchers are excluded.
- Router threshold calibration moves the whole model to one GPU before evaluation, even with `--multigpu`.
- Residual experts use floating-point matrix multiplications with masked inputs. This code does not establish packed low-bit memory savings or sparse compute speedups.
- The tested smoke model is synthetic and small. No pretrained 1B/3B/8B PPL or accuracy result is claimed.

## Migration and validation

`MIGRATION_MANIFEST.json` records source commit and SHA-256 for every copied file. Core source, model/quantization helpers, bundled `lm_eval`, license and relevant BitMoE shell experiments are included. Models/quantization directories retain compatibility variants because the entry imports OPT/Falcon modules as well as LLaMA helpers.

Excluded: untracked working files, `.aws`, `.codex`, git history, credentials, pretrained weights, datasets/cache, experiment outputs, notebooks/visualizations, legacy OmniQuant entrypoints and unrelated/broken legacy shell launchers. No remote or source file was changed in the original checkout.

`validation/gpu_smoke.py` validates both training stages with backward passes, routed inference across all seven projections, actual layer-wise calibration on a complete synthetic LLaMA model, three checkpoint files, restore, threshold calibration and matching restored logits. Results are in `validation/gpu-smoke.log`. Python syntax, shell syntax and source hashes are checked separately.
