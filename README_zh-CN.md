<p align="center">
  <img src="assets/mobiquant-banner.png" alt="MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM" width="100%">
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-374151?style=for-the-badge" alt="English"></a>
  &nbsp;
  <a href="README_zh-CN.md"><img src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-2563EB?style=for-the-badge" alt="简体中文"></a>
</p>


本仓库是论文 **MoBiQuant: Mixture-of-Bits Quantization for
Token-Adaptive Any-Precision LLM** 的官方实现（[论文链接](https://arxiv.org/abs/2602.20191)）。

<p align="center">
  <a href="https://arxiv.org/abs/2602.20191"><img src="https://img.shields.io/badge/arXiv-2602.20191-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="在 arXiv 上阅读论文"></a>
  &nbsp;
  <a href="https://huggingface.co/Simwang/MoBiQuant"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoints-333333?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="从 Hugging Face 下载模型参数"></a>
  &nbsp;
  <a href="https://simwangarizona.github.io/MoBiQuant_Page/"><img src="https://img.shields.io/badge/Project-Page-2563EB?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="访问项目主页"></a>
</p>


## 最新动态

- **2026/10/16** — 代码和[模型参数](https://huggingface.co/Simwang/MoBiQuant)已发布，[项目主页](https://simwangarizona.github.io/MoBiQuant_Page/)已上线！
- **2026/09/24** — MoBiQuant 被 **NeurIPS** 接收！🎉

## 1. 安装依赖

### 创建环境

环境要求：**Python 3.10.18**、**PyTorch 2.8.0 + CUDA 12.8** 和 **Transformers 4.56.1**。

```bash
git clone https://github.com/SimWangArizona/MoBiQuant.git
cd MoBiQuant

conda create -n mobiquant python=3.10.18 -y
conda activate mobiquant

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

### 检查 CUDA 并登录 Hugging Face

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
hf auth login
```


## 2. 下载原始模型


### 设置模型目录并下载

**请将下方路径替换为你的模型目录。** 

```bash
export MOBIQUANT_MODEL_ROOT="$PWD/pretrained_models"
# 如果要复用已有的本地模型，可设置为：
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

只需下载你要使用的模型。这里固定的版本与原始本地模型及 tokenizer 配置一致，详见[模型配置](workflows/_config.py)。

## 3. 复现论文评估结果

### 运行细粒度 sweep

请先将原始基础模型准备到第 2 节设置的目录中。

各模型脚本会自动从 **[Simwang/MoBiQuant](https://huggingface.co/Simwang/MoBiQuant)** 下载对应的 `omni_parameters.pth` 和 `router_parameters.pth`，然后运行 WikiText2 评估。

```bash
bash scripts/llama2-7b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-8b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-1b/finegrained-sweep-eval.sh --gpu 0
bash scripts/llama3-3b/finegrained-sweep-eval.sh --gpu 0
```

运行你要评估的模型所对应的命令。每次 sweep 覆盖 **2.0–8.0 的目标 bit 数**。MoBiQuant 计算的是 token 间的平均位宽，因此支持小数位宽。例如，75% 的 token 使用 4-bit 精度，其余 25% 使用 6-bit 精度，平均位宽就是 4.5 bits。


首次运行时会自动下载 WikiText2，随后即可复现论文中的评估结果。你也可以查看 `eval.log`，了解 token 分配统计及各线性投影的详细信息。

### 使用自定义路径运行

```bash
# 将 /path/to/base/model 替换为对应原始模型所在的目录。
# 将 /my/checkpoints 替换为本地模型参数的下载根目录。

bash scripts/llama3-1b/finegrained-sweep-eval.sh \
  --model /path/to/base/model \
  --checkpoint-root /my/checkpoints \
  --output results/custom-llama3-1b \
  --gpu 0
```

## 4. 训练自己的模型参数

### 从原始基础模型开始训练

训练从原始预训练模型开始。
训练完成后，各模型的 `train.sh` 会自动评估**本次训练生成的参数**。 

```bash
bash scripts/llama2-7b/train.sh --gpu 0
bash scripts/llama3-8b/train.sh --gpu 0
bash scripts/llama3-1b/train.sh --gpu 0
bash scripts/llama3-3b/train.sh --gpu 0
```

如果只想训练，可以运行：

```bash
# 仅训练并保存参数，不自动运行 sweep：
bash scripts/llama3-1b/train.sh --gpu 0 --skip-sweep
```

详细训练配置见 [training_recipes.json](workflows/training_recipes.json)。


默认训练输出目录：`log/retrained-<preset>/`。
默认 sweep 输出目录：训练输出目录下的 `finegrained-sweep-eval/`。

训练会保存 `omni_parameters.pth`、`router_parameters.pth` 和 `router_stats.pth`。其中，`router_stats.pth` 保存训练配置和恢复训练所需的 router 温度调度总步数；评估时该文件为可选项。



## 开源进度

- ✅ 算法与伪量化（fake quantization）
- ⬜ 计算内核与打包后的低比特模型参数（packed checkpoints）

## 致谢

本项目基于 [OmniQuant](https://github.com/OpenGVLab/OmniQuant) 的开源代码实现，感谢其作者分享相关工作。

## 欢迎引用

如果 MoBiQuant 对你的研究有所帮助，欢迎引用我们的论文：

```bibtex
@article{wang2026mobiquant,
  title={MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Any-Precision LLM},
  author={Wang, Dongwei and Kim, Jinhee and Han, Seokho and Gudovskiy, Denis and Nakata, Yohei and Okuno, Tomoyuki and Peong, KhayTze and Jeon, Kang Eun and Ko, Jong Hwan and Chen, Yiran and others},
  journal={arXiv preprint arXiv:2602.20191},
  year={2026}
}
```
