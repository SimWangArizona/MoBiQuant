"""Load the original LLaMA weights used by calibration and WikiText2 evaluation."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class LMClass:
    def __init__(self, args):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = AutoConfig.from_pretrained(args.model, attn_implementation=args.attn_implementation)
        if config.model_type != 'llama':
            raise ValueError('The supported workflows require a LLaMA base model')
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, device_map='cpu', torch_dtype=torch.float16,
        )
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()

    @property
    def device(self):
        return self._device
