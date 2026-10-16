import torch
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_moe import QuantMoE
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.omni_norm import OmniLlamaRMSNorm
from collections import OrderedDict
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,LlamaRMSNorm,repeat_kv, LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import pdb
import copy
from models.transformation import *




class QuantLlamaMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        # self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = QuantLinear(org_module.gate_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params)
        self.down_proj = QuantLinear(org_module.down_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params)
        self.up_proj = QuantLinear(org_module.up_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuantLlamaMoE(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        args=None,
    ):
        super().__init__()
        print(f"QuantllamaMLPMoE {args.weight_quant_params}")

        self.gate_proj = QuantMoE(org_module.gate_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params,
                                           args.slice_bits_list,
                                           args.global_steps)
        self.down_proj = QuantMoE(org_module.down_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params,
                                           args.slice_bits_list,
                                           args.global_steps)
        self.up_proj = QuantMoE(org_module.up_proj,
                                           args.weight_quant_params,
                                           args.act_quant_params,
                                           args.slice_bits_list,
                                           args.global_steps)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x, step=None):
        return self.down_proj(self.act_fn(self.gate_proj(x, step)) * self.up_proj(x, step), step)


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, 
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # self.rotary_emb = copy.deepcopy(org_module.rotary_emb)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        # print(f"QuantllamaAttention {args.weight_quant_params}")
        # 临时修改nbit为attnbits
        wq_params = copy.deepcopy(args.weight_quant_params)
        wq_params["n_bits"] = args.attnbits   # 只改这一份
        print(f"Attention block quantization bits: {wq_params['n_bits']}")
        self.k_proj = QuantLinear(
            org_module.k_proj,
            wq_params,
            args.act_quant_params,
        )
        self.v_proj = QuantLinear(
            org_module.v_proj,
            wq_params,
            args.act_quant_params,
        )
        self.q_proj = QuantLinear(
            org_module.q_proj,
            wq_params,
            args.act_quant_params,
        )
        self.o_proj = QuantLinear(
            org_module.o_proj, wq_params, args.act_quant_params
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states =self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and len(past_key_value) > 0 and hasattr(past_key_value[0], 'shape'):
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        # [bsz, nh, t, hd]

        if past_key_value is not None and len(past_key_value) > 0 and hasattr(past_key_value[0], 'shape'):
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)
                

class QuantLlamaAttentionMoE(nn.Module):
    """Multi-headed attention with MoE quantization from 'Attention Is All You Need' paper"""

    def __init__(self,
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # self.rotary_emb = copy.deepcopy(org_module.rotary_emb)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.k_proj = QuantMoE(
            org_module.k_proj,
            args.weight_quant_params,
            args.act_quant_params,
            args.slice_bits_list,
            args.global_steps,
        )
        self.v_proj = QuantMoE(
            org_module.v_proj,
            args.weight_quant_params,
            args.act_quant_params,
            args.slice_bits_list,
            args.global_steps,
        )
        self.q_proj = QuantMoE(
            org_module.q_proj,
            args.weight_quant_params,
            args.act_quant_params,
            args.slice_bits_list,
            args.global_steps,
        )
        self.o_proj = QuantMoE(
            org_module.o_proj,
            args.weight_quant_params,
            args.act_quant_params,
            args.slice_bits_list,
            args.global_steps,
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        self.use_weight_quant = False
        self.use_act_quant = False

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        step: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # query_states = self.q_proj(hidden_states, step)
        # key_states = self.k_proj(hidden_states, step)
        # value_states = self.v_proj(hidden_states, step)
        query_states = self.q_proj(hidden_states, step).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states, step).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states, step).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and len(past_key_value) > 0 and hasattr(past_key_value[0], 'shape'):
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        # [bsz, nh, t, hd]

        if past_key_value is not None and len(past_key_value) > 0 and hasattr(past_key_value[0], 'shape'):
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = self.qkt_matmul.quant_x1(query_states)
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.pv_matmul.quant_x1(attn_weights)
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output, step)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantMoE)):
                m.set_quant_state(weight_quant, act_quant)



class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 ori_layer,
                 args):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Conditionally use MoE for attention projections
        if args.slice_bits_list is not None:
            print("Using QuantLlamaAttention/QuantLlamaMoE!!")
            self.self_attn = QuantLlamaAttention(
                org_module=ori_layer.self_attn,
                config=config,
                args=args,
            )
            self.mlp = QuantLlamaMoE(
                org_module=ori_layer.mlp,
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                args=args,
            )
        else:
            print("Using QuantLlamaAttention/QuantLlamaMLP!")
            self.self_attn = QuantLlamaAttention(
                org_module=ori_layer.self_attn,
                config=config,
                args=args,
            )
            self.mlp = QuantLlamaMLP(
                org_module=ori_layer.mlp,
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                args=args,
            )

        self.input_layernorm = OmniLlamaRMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = OmniLlamaRMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            step (`int`, *optional*): training step for MoE router scheduling
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)


        # Self Attention
        if step is None or isinstance(self.self_attn, QuantLlamaAttention):
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                step=step,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)


        if step is None or isinstance(self.mlp, QuantLlamaMLP):
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states, step=step)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # if only return hidden states, remove tuple
        if len(outputs) == 1:
            return outputs[0]

        return outputs        

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul, QuantMoE)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)
      
    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.mlp.down_proj.temp_weight = self.mlp.down_proj.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, (QuantLinear, QuantMoE)):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, (QuantLinear, QuantMoE)):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.input_layernorm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.post_attention_layernorm,[self.mlp.up_proj,self.mlp.gate_proj],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.o_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, (QuantLinear, QuantMoE)):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, (QuantLinear, QuantMoE)):
                module.weight_quantizer.register_scales_and_zeros()

    def get_layer_reg_loss(self, args, step=None):
        """Compute layer-wise regularization loss by aggregating all QuantMoE modules.

        Returns:
            reg_loss: Regularization loss for this layer
            module_stats: Dict containing per-module statistics
        """
        # Collect all QuantMoE modules in this layer
        layer_modules = []
        module_sparsity = {}
        module_eff_bits = {}

        for name, module in self.named_modules():
            if isinstance(module, QuantMoE):
                layer_modules.append(module)
                module_sparsity[name] = module.get_sparsity()
                module_eff_bits[name] = module.get_effective_bits()

        if not layer_modules:
            # No QuantMoE modules, return zero loss
            zero = torch.tensor(0.0, device=next(self.parameters()).device, dtype=next(self.parameters()).dtype)
            return zero, {'sparsity': {}, 'eff_bits': {}}

        # Compute layer-wise regularization loss
        ref_module = layer_modules[0]
        device = ref_module.Mask_Router.scalar.device
        dtype = ref_module.Mask_Router.scalar.dtype
        zero = torch.tensor(0.0, device=device, dtype=dtype)

        if args.target_activation_ratio is not None:
            # Mode 1: Regularize based on target activation ratio
            target_ratio = args.target_activation_ratio
            if step is not None:
                total_steps = ref_module.Mask_Router.global_step
                initial_ratio = 0.9
                ratio_log = math.log(max(1, step)) / math.log(total_steps)
                target_ratio = initial_ratio - (initial_ratio - target_ratio) * ratio_log

            total_elements = 0
            current_activated = zero
            sum_l1 = zero
            for module in layer_modules:
                from quantize.int_moe import hard_gate
                router_scalar = module.Mask_Router.scalar
                total_elements += router_scalar.numel()
                current_activated = current_activated + hard_gate(router_scalar.detach(), threshold=0.5).sum()
                sum_l1 = sum_l1 + torch.norm(router_scalar, p=1)
            target_activated = target_ratio * total_elements
            diff = (current_activated - target_activated) / target_activated
            reg_loss = diff * sum_l1

        elif args.target_bit is not None:
            # Mode 2: Regularize based on target effective bits
            target_bit = args.target_bit
            if step is not None:
                total_steps = ref_module.Mask_Router.global_step
                if len(ref_module.bit_list) > 0:
                    initial_bit = float(sum(ref_module.bit_list))
                else:
                    initial_bit = 8.0
                ratio_log = math.log(max(1, step)) / math.log(total_steps)
                target_bit_scheduled = initial_bit - (initial_bit - target_bit) * ratio_log
            else:
                target_bit_scheduled = target_bit

            sum_l1 = zero
            current_eff_bits = 0.0
            for module in layer_modules:
                current_eff_bits += module.get_effective_bits()
                sum_l1 = sum_l1 + torch.norm(module.Mask_Router.scalar, p=1)
            current_eff_bits /= max(1, len(layer_modules))
            diff = (current_eff_bits - target_bit_scheduled) / target_bit_scheduled
            reg_loss = diff * sum_l1

        else:
            # No regularization
            reg_loss = zero

        module_stats = {
            'sparsity': module_sparsity,
            'eff_bits': module_eff_bits
        }

        return reg_loss, module_stats