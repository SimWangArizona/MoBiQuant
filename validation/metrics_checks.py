"""PPL normalization regression: uniform logits must yield vocabulary-size PPL."""
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from main_bitmoe import evaluate
config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
config._attn_implementation='eager'
model=LlamaForCausalLM(config).eval()
with torch.no_grad():
    for p in model.parameters(): p.zero_()
lm=SimpleNamespace(model=model,device=torch.device('cpu'),seqlen=8)
args=SimpleNamespace(multigpu=False,net='Llama-test',tasks='',seed=2,model='synthetic/Llama',limit=1,wandb=False)
loader=SimpleNamespace(input_ids=torch.arange(32).reshape(1,-1))
with patch('main_bitmoe.get_loaders', return_value=([],loader)):
    for limit in [1,2,-1]:
        args.limit=limit
        result=evaluate(lm,args,logging.getLogger('ppl-test'))
        assert abs(result['wikitext2']-32)<1e-4,result
print('PASS uniform-logit PPL normalization with 1, 2 and all evaluation sequences')
