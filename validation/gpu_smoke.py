"""Offline synthetic GPU smoke check; no pretrained weights or datasets required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from parser import args
from models.int_llama_layer import QuantLlamaDecoderLayer
from quantize.int_moe import QuantMoE
from quantize.utils import compute_stage_config, smooth_and_quant_temporary, clear_temp_variable, smooth_and_quant_inplace

torch.manual_seed(2)
args.wbits=2; args.abits=16; args.slice_bits_list=[2,2,2,2]
args.global_steps=10; args.onlymlp=False; args.let=False; args.target_bit_eval=4.0
args.weight_quant_params=dict(n_bits=2,per_channel_axes=[0],symmetric=False,dynamic_method='per_channel',group_size=128,lwc=True,disable_zero_point=False)
args.act_quant_params=dict(n_bits=16,per_channel_axes=[],symmetric=False,dynamic_method='per_token')
args.q_quant_params=args.k_quant_params=args.v_quant_params=args.act_quant_params
args.p_quant_params=dict(n_bits=16,metric='fix0to1')
config=LlamaConfig(hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32)
config._attn_implementation='eager'
original=LlamaDecoderLayer(config,layer_idx=0).cuda()
layer=QuantLlamaDecoderLayer(config,original,args).cuda()
modules=[m for m in layer.modules() if isinstance(m,QuantMoE)]
assert len(modules)==7, len(modules)
x=torch.randn(1,8,128,device='cuda'); pos=torch.arange(8,device='cuda').unsqueeze(0)
rope=LlamaRotaryEmbedding(config).cuda(); embeddings=rope(x,pos)
mask=torch.full((1,1,8,8),torch.finfo(x.dtype).min,device='cuda').triu(1)
for stage in compute_stage_config(args):
    smooth_and_quant_temporary(layer,args,True,buildq_mode=True,stage_cfg=stage)
    out=layer(x,attention_mask=mask,position_ids=pos,position_embeddings=embeddings,step=1)
    assert out.shape==x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    layer.zero_grad(set_to_none=True)
    print('PASS training forward/backward:',stage['stage_type'])
clear_temp_variable(layer)
smooth_and_quant_inplace(layer,args,True)
layer.eval()
with torch.no_grad():
    out=layer(x,attention_mask=mask,position_ids=pos,position_embeddings=embeddings)
assert out.shape==x.shape and torch.isfinite(out).all()
print('PASS routed inference: 7 Attention/MLP projections; shape',tuple(out.shape))
print('GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__)

# Exercise the actual layer-wise calibration and directory checkpoint restore.
import tempfile
import logging
from types import SimpleNamespace
from transformers import LlamaForCausalLM
from quantize.bitmoequant_llama3 import bitmoequant
from main_bitmoe import calibrate_router_threshold
args.net='Llama-smoke'; args.model='synthetic/Llama-smoke'; args.epochs=1
args.nsamples=2; args.batch_size=1; args.wandb=False; args.resume=None
args.target_activation_ratio=0.33; args.target_activation_ratio_eval=None
args.real_quant=False
config.vocab_size=256; config.use_cache=False
logger=logging.getLogger('smoke'); logger.addHandler(logging.StreamHandler()); logger.setLevel(logging.INFO)
def make_lm():
    torch.manual_seed(4)
    model=LlamaForCausalLM(config).half().eval()
    model.requires_grad_(False)
    return SimpleNamespace(model=model,device=torch.device('cuda:0'),_device='cuda:0',seqlen=8)
data=[(torch.randint(0,256,(1,8)),None) for _ in range(2)]
with tempfile.TemporaryDirectory(prefix='mobiquant-checkpoint-') as checkpoint:
    args.output_dir=checkpoint
    lm=make_lm()
    bitmoequant(lm,args,data,None,None,logger)
    for name in ['omni_parameters.pth','router_parameters.pth','router_stats.pth']:
        assert (Path(checkpoint)/name).is_file(), name
    calibrate_router_threshold(lm,args,data,logger)
    with torch.no_grad(): reference=lm.model(data[0][0].cuda()).logits
    assert torch.isfinite(reference).all()
    args.epochs=0; args.resume=checkpoint
    restored=make_lm()
    bitmoequant(restored,args,data,None,None,logger)
    calibrate_router_threshold(restored,args,data,logger)
    with torch.no_grad(): actual=restored.model(data[0][0].cuda()).logits
    torch.testing.assert_close(actual,reference,atol=1e-3,rtol=1e-3)
    print('PASS full synthetic model: layer-wise calibration, checkpoint save/restore, threshold calibration, matching logits')
