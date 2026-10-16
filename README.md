# MoBiQuant

Selected BitMoE training and evaluation code from BitMoEV2 branch `buildq2`, commit `7d493d8ffe7a4367b8f99375a9599547dab5ce99`. The original checkout is untouched. `MIGRATION_MANIFEST.json` records initial import hashes; targeted fixes made only in this destination repository are described in `validation/FEATURE_AUDIT.md`.

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

For a new environment, install a suitable CUDA PyTorch build and the project dependencies from `pyproject.toml`; `termcolor` and the validated Transformers 4.56.1 pin are now included in project dependencies. Use `pip install -r requirements-validated.lock` for the tested direct package versions, selecting a suitable PyTorch CUDA build for the machine. Transitive dependencies are resolved by pip; this is not a full environment lock. AutoGPTQ is optional for this floating-point simulation workflow; its absence prints a message. Real packed quantization has not been validated.

## Recommended four-model workflow

Use `scripts/bitmoe/run.py` for the audited workflow. Presets are `llama2-7b`, `llama3-1b` (Llama-3.2-1B), `llama3-3b` (Llama-3.2-3B base), and `llama3-8b` (Llama-3-8B). Each trains with `--target_bit 3`, all Attention and MLP projections in MoE mode, and saves to `log/<preset>-target3` by default. This is a training objective, not a guarantee that learned routing reaches exactly 3 bits.

```bash
conda activate omniquant
python scripts/bitmoe/run.py train llama2-7b --gpu 0
python scripts/bitmoe/run.py sweep llama2-7b --gpu 0
```

Shell equivalents are `bash scripts/bitmoe/train_target3.sh llama2-7b --gpu 0` and `bash scripts/bitmoe/sweep_eval.sh llama2-7b --gpu 0`. Replace `llama2-7b` with any other preset for the same workflow. `--model`, `--checkpoint`, `--output`, `--epochs`, `--nsamples`, `--seqlen` and `--limit` are configurable. Use `--dry-run` to inspect commands. Sweep defaults to 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8 bits; use `--bits 2 3 4 5 6 7 8` to choose points.

Results are written incrementally to `results/<preset>/sweep/bits-performance.csv` with `target_bit_eval,actual_bits,wikitext2_ppl,moe_modules,status,metrics_file,log_file`. Each point has its own `metrics.json` and `eval.log`. Runs fail on subprocess errors or missing/invalid metrics, preserve completed rows, and refuse to overwrite nonempty output directories. Metrics come from JSON, not grep of a shared log. Actual bits are the mean over calibration samples and MoE projections (not a parameter-weighted storage bit rate). The 2-bit endpoint disables residual experts and records exactly 2 bits.

The shell scripts below are retained historical experiments, not the recommended four-model workflow. In particular the legacy 1B script named w3 actually targets about 4.1 bits, and old sweeps have mismatched checkpoint paths and fragile log extraction.

## Low-level train and evaluate

Run from the repository root. Replace model, GPU and output paths as needed. Original model weights and Hugging Face dataset access/cache are required for real model runs.

```bash
CUDA_VISIBLE_DEVICES=0 python main_bitmoe.py \
  --model /mnt/fast7T/models/Llama-3-8B \
  --output_dir ./log/Llama-3-8B-bitmoe \
  --wbits 2 --abits 16 --group_size 128 \
  --slice_bits_list 2 2 2 2 --stage_weight 1.0 1.33 \
  --epochs 20 --batch_size 1 --lwc --lwc_lr 1e-3 \
  --bitmoe_lr 8e-5 --reg_weight 8e-5 \
  --target_bit 3 --eval_ppl
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
- The tested smoke model is synthetic and small. The feature audit runs reduced real-weight WikiText checks separately; these do not establish production training convergence or full-test-set PPL.

## Migration and validation

`MIGRATION_MANIFEST.json` records source commit and SHA-256 for every copied file. Core source, model/quantization helpers, bundled `lm_eval`, license and relevant BitMoE shell experiments are included. Models/quantization directories retain compatibility variants because the entry imports OPT/Falcon modules as well as LLaMA helpers.

Excluded: untracked working files, `.aws`, `.codex`, git history, credentials, pretrained weights, datasets/cache, experiment outputs, notebooks/visualizations, legacy OmniQuant entrypoints and unrelated/broken legacy shell launchers. No remote or source file was changed in the original checkout.

`validation/gpu_smoke.py` validates both training stages with backward passes, routed inference across all seven projections, actual layer-wise calibration on a complete synthetic LLaMA model, three checkpoint files, restore, threshold calibration and matching restored logits. Results are in `validation/gpu-smoke.log`. Python syntax, shell syntax and source hashes are checked separately.
