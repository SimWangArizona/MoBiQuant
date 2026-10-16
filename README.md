<p align="center">
  <img src="assets/mobiquant-banner.png" alt="MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM" width="100%">
</p>

This is the official implementation of **MoBiQuant: Mixture-of-Bits Quantization for
Token-Adaptive Any-Precision LLM** ([paper](https://arxiv.org/abs/2602.20191)).

<p align="center">
  <a href="https://arxiv.org/abs/2602.20191"><img src="https://img.shields.io/badge/arXiv-2602.20191-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Read the paper on arXiv"></a>
  &nbsp;
  <a href="https://huggingface.co/Simwang/MoBiQuant"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoints-333333?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Download checkpoints on Hugging Face"></a>
  &nbsp;
  <a href="https://simwangarizona.github.io/MoBiQuant_Page/"><img src="https://img.shields.io/badge/Project-Page-2563EB?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Visit the project page"></a>
</p>


## News

- **2026/10/16** — Code and [checkpoints](https://huggingface.co/Simwang/MoBiQuant) released; [project page](https://simwangarizona.github.io/MoBiQuant_Page/) is online!
- **2026/09/24** — MoBiQuant accepted to **NeurIPS**! 🎉

## 1. Install dependencies

### Create the environment

Tested with **Python 3.10.18**, **PyTorch 2.8.0 + CUDA 12.8**, and **Transformers 4.56.1**.
Run all commands from the repository root.

```bash
git clone https://github.com/SimWangArizona/MoBiQuant.git
cd MoBiQuant

conda create -n mobiquant python=3.10.18 -y
conda activate mobiquant

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

[requirements.txt](requirements.txt) pins the runtime packages from the working `omniquant` environment. It excludes unrelated tools installed in that environment.
The CUDA wheel follows the [official PyTorch installation command](https://pytorch.org/get-started/previous-versions/#v280).

### Check CUDA and log in

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
hf auth login
```

A CUDA GPU is required for training and evaluation. Accept the relevant Meta model access terms on Hugging Face before downloading gated base models. AutoGPTQ and W&B are not required for these commands.

## 2. Download the original models

### Exact model variants

Use these **Base** models. The presets do not use Instruct models or Llama-3.1.

| Preset | Original Hugging Face model | Local folder under `MODEL_ROOT` |
|---|---|---|
| `llama2-7b` | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | `Llama-2-7b-hf` |
| `llama3-8b` | [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | `Llama-3-8B` |
| `llama3-1b` | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | `Llama-3.2-1B` |
| `llama3-3b` | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) | `Llama-3.2-3B` |

### Choose the model location and download

**Replace the path below with your model directory.** Existing local models can be reused with the folder names above.

```bash
export MOBIQUANT_MODEL_ROOT="$PWD/pretrained_models"
# To reuse the original local models instead:
# export MOBIQUANT_MODEL_ROOT="/mnt/fast7T/models"

hf download meta-llama/Llama-2-7b-hf \
  --revision 01c7f73d771dfac7d292323805ebc428287df4f9 \
  --local-dir "$MOBIQUANT_MODEL_ROOT/Llama-2-7b-hf" \
  --include "*.safetensors" "*.json" "tokenizer.model"

hf download meta-llama/Meta-Llama-3-8B \
  --revision 8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
  --local-dir "$MOBIQUANT_MODEL_ROOT/Llama-3-8B" \
  --include "*.safetensors" "*.json"

hf download meta-llama/Llama-3.2-1B \
  --revision 4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --local-dir "$MOBIQUANT_MODEL_ROOT/Llama-3.2-1B" \
  --include "*.safetensors" "*.json"

hf download meta-llama/Llama-3.2-3B \
  --revision 13afe5124825b4f3751f836b40dafda64c1ed062 \
  --local-dir "$MOBIQUANT_MODEL_ROOT/Llama-3.2-3B" \
  --include "*.safetensors" "*.json"
```

Download only the models you need. These pinned revisions match the original local model and tokenizer configurations; see [model configuration](workflows/_config.py).

## 3. Reproduce the paper evaluation

### Run a fine-grained sweep

Each model's script automatically downloads its `omni_parameters.pth` and `router_parameters.pth` from **[Simwang/MoBiQuant](https://huggingface.co/Simwang/MoBiQuant)**, then runs WikiText2 evaluation.
The original base model must already be available at the location configured in Section 2.

```bash
bash scripts/llama2-7b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-8b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-1b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-3b/finegrained-sweep-eval.sh --gpu 0
```

Run the command for the model you want to evaluate. Each sweep covers **2.0–8.0 target bits in steps of 0.1: 61 points**.

| Setting | Value |
|---|---|
| Calibration dataset | WikiText2 |
| Calibration samples | 128 |
| Seed | 2 |
| Sequence length | 2048 |
| Test evaluation | Full (`--limit -1`) |
| Parameter download | `checkpoints/<preset>/` |

WikiText2 is downloaded automatically on first use. Each point reloads the published checkpoint and recalibrates routing as applicable. Evaluation requires only the two parameter files, without `router_stats.pth`.

### Override paths or preview the sweep

```bash
# Replace /path/to/base/model with the matching original model directory.
bash scripts/llama3-1b/finegrained-sweep-eval.sh \
  --model /path/to/base/model \
  --checkpoint-root checkpoints \
  --output results/custom-llama3-1b \
  --gpu 0

# Preview all 61 commands without downloading or running evaluation:
bash scripts/llama3-1b/finegrained-sweep-eval.sh --dry-run
```

### Read the results

Each model writes `results/<preset>/finegrained-sweep-eval/`:

| File | Contents |
|---|---|
| `bits-performance.csv` | Target bits, actual bits, WikiText2 PPL, and run status |
| `bits-<target>/metrics.json` | Per-point metrics |
| `bits-<target>/eval.log` | Per-point log |
| `sweep-plan.json` | Download revision and all evaluation commands |

Use **actual bits** when comparing bit–PPL curves. This statistic averages over projections on the final calibration sample; it is not a parameter-weighted storage estimate.
Existing nonempty output directories are not overwritten; choose a new `--output` for another run.

## 4. Train your own parameters

### Train from the original base model

Training starts from the original pretrained model, **without loading MoBiQuant `.pth` files or resuming a checkpoint**.
These presets use the training recipes recorded in the released checkpoints' logs.
After training succeeds, each `train.sh` automatically evaluates its **newly trained parameters** at **2.0–8.0 target bits, step 0.1 (61 points)**. This sweep uses the saved local checkpoint without downloading the published HF parameters.

```bash
bash scripts/llama2-7b/train.sh --gpu 0
bash scripts/llama3-8b/train.sh --gpu 0
bash scripts/llama3-1b/train.sh --gpu 0
bash scripts/llama3-3b/train.sh --gpu 0
```

Run the command for the model you want to train. All recipes use WikiText2, 128 samples, batch size 1, 20 epochs, group size 128, and bit slices `[2, 2, 2, 2]`.
Attention q/k/v/o and MLP gate/up/down projections use routed residual slices.

| Preset | Router LR | LWC LR | Regularization weight | Stage weights | Target activation ratio |
|---|---:|---:|---:|---|---:|
| `llama2-7b` | 5e-6 | 1e-2 | 1e-5 | `[1.0, 1.3]` | 0.16 |
| `llama3-8b` | 5e-6 | 1e-2 | 1e-5 | `[1.0, 1.3]` | 0.16 |
| `llama3-1b` | 8e-5 | 1e-3 | 8e-5 | `[1.0, 1.33]` | 0.15 |
| `llama3-3b` | 4e-5 | 1e-3 | 4e-5 | `[1.0, 1.33]` | 0.15 |

### Choose an output location

```bash
bash scripts/llama3-1b/train.sh \
  --model /path/to/base/model \
  --output log/my-llama3-1b \
  --sweep-output results/my-llama3-1b \
  --gpu 0

# Inspect both the training and post-training sweep commands:
bash scripts/llama3-1b/train.sh --dry-run
```

Default training output: `log/retrained-<preset>/`.
Default sweep output: `<training output>/finegrained-sweep-eval/`, including `bits-performance.csv` and per-point metrics/logs.
The overall sweep log is `<training output>/sweep.log`.

Training saves `omni_parameters.pth`, `router_parameters.pth`, `router_stats.pth`, the selected recipe, and logs. Original base-model weights are kept separately.

### Run only training or reevaluate saved parameters

```bash
# Save parameters without automatically running the sweep:
bash scripts/llama3-1b/train.sh --gpu 0 --skip-sweep

# Run the same 61-point sweep on an existing trained checkpoint:
bash scripts/llama3-1b/finegrained-sweep-eval.sh \
  --checkpoint log/retrained-llama3-1b \
  --output results/retrained-llama3-1b \
  --gpu 0
```

If training fails or parameter files are missing, the automatic sweep does not start. If evaluation fails, the trained checkpoint is preserved; rerun evaluation with a new `--output` directory.

Exact training settings and source log names are retained in [training_recipes.json](workflows/training_recipes.json).

The `scripts/` directory contains exactly these two launchers per model. Their shared implementation lives in `workflows/`.

## Open Source TODO

- ✅ Algorithm and fake quantization
- ⬜ Kernels and packed checkpoints
