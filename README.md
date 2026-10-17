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

Need **Python 3.10.18**, **PyTorch 2.8.0 + CUDA 12.8**, and **Transformers 4.56.1**.

```bash
git clone https://github.com/SimWangArizona/MoBiQuant.git
cd MoBiQuant

conda create -n mobiquant python=3.10.18 -y
conda activate mobiquant

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

### Check CUDA and log in

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
hf auth login
```


## 2. Download the original models


### Choose the model location and download

**Replace the path below with your model directory.** 

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

The original base model must already be available at the location configured in Section 2.

Each model's script automatically downloads its `omni_parameters.pth` and `router_parameters.pth` from **[Simwang/MoBiQuant](https://huggingface.co/Simwang/MoBiQuant)**, then runs WikiText2 evaluation.

```bash
bash scripts/llama2-7b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-8b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-1b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-3b/finegrained-sweep-eval.sh --gpu 0
```

Run the command for the model you want to evaluate. Each sweep covers **2.0–8.0 target bits**. Fractional bit widths are possible because MoBiQuant computes the average bit width across tokens. For example, assigning 4-bit precision to 75% of the tokens and 6-bit precision to the remaining 25% yields an average bit width of 4.5 bits.


WikiText2 is downloaded automatically on first use. You can then reproduce the results reported in the paper. You can also inspect `eval.log` for token assignment statistics and detailed information about each linear projection.

### Run with custom paths

```bash
# Replace /path/to/base/model with the directory containing the matching original model.
# Replace /my/checkpoints with your local checkpoint download root.

bash scripts/llama3-1b/finegrained-sweep-eval.sh \
  --model /path/to/base/model \
  --checkpoint-root /my/checkpoints \
  --output results/custom-llama3-1b \
  --gpu 0
```

## 4. Train your own parameters

### Train from the original base model

Training starts from the original pretrained model.
After training, each `train.sh` automatically evaluates its **newly trained parameters**. 

```bash
bash scripts/llama2-7b/train.sh --gpu 0
bash scripts/llama3-8b/train.sh --gpu 0
bash scripts/llama3-1b/train.sh --gpu 0
bash scripts/llama3-3b/train.sh --gpu 0
```

You can also run training-only by:

```bash
# Save parameters without automatically running the sweep:
bash scripts/llama3-1b/train.sh --gpu 0 --skip-sweep
```

The detailed training recipes are in [training_recipes.json](workflows/training_recipes.json).


Default training output: `log/retrained-<preset>/`.
Default sweep output: `<training output>/finegrained-sweep-eval/`.

Training saves `omni_parameters.pth`, `router_parameters.pth`, `router_stats.pth`. The `router_stats.pth` file stores training configuration and the router temperature schedule length needed when resuming training; it is optional for evaluation.



## Open Source TODO

- ✅ Algorithm and fake quantization
- ⬜ Kernels and packed checkpoints

## Acknowledgements

Our implementation builds on the open-source code from [OmniQuant](https://github.com/OpenGVLab/OmniQuant). We thank its authors for sharing their work.

## Welcome to cite us

If you find MoBiQuant useful in your research, please consider citing our paper:

```bibtex
@article{wang2026mobiquant,
  title={MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM},
  author={Wang, Dongwei and Kim, Jinhee and Han, Seokho and Gudovskiy, Denis and Nakata, Yohei and Okuno, Tomoyuki and Peong, KhayTze and Jeon, Kang Eun and Ko, Jong Hwan and Chen, Yiran and others},
  journal={arXiv preprint arXiv:2602.20191},
  year={2026}
}
```
