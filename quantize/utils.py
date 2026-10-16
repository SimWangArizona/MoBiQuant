from collections import OrderedDict
from quantize.int_linear import QuantLinear
import torch
from quantize.int_matmul import QuantMatMul
from quantize.int_moe import QuantMoE
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def bitmoe_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('router') > -1:
            params.append(m)
    return iter(params)

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def get_bitmoe_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find('router') > -1:
            # print(f"bound_factor: {n}" if n.find('bound_factor') > -1 else "", end="")
            # print(f", smooth: {n}" if n.find(template) > -1 else "", end="")
            # print(f", router: {n}" if n.find('router') > -1 else "")
            params.append(m)
    return iter(params)  

def find_grad_norm(model, use_shift=True):
    grad_info = []
    template = "smooth" if use_shift else "smooth_scale"

    for n, m in model.named_parameters():
        if (
            'bound_factor' in n 
            or template in n 
            or 'router' in n
        ):
            if m.grad is not None:
                g = m.grad.norm().item()
                grad_info.append((n, g))
            else:
                grad_info.append((n, None))

    return grad_info

def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantMoE)):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama, buildq_mode=False, stage_cfg=None):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            if not args.onlymlp:
                smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                        model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            if not args.onlymlp:
                smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                    model.out_smooth_scale, model.out_smooth_shift)
                smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                    model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, (QuantLinear, QuantMoE)):
                module.temp_weight = module.weight
    # Quantization (modified to handle BuildQ mode)
    for name, module in model.named_modules():
        if isinstance(module, (QuantMoE, QuantLinear)):
            if not hasattr(module, "temp_weight"):
                module.temp_weight = module.weight
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias

            if isinstance(module, QuantMoE) and buildq_mode:
                # BuildQ mode: store stage config, quantization happens in QuantMoE forward
                _ = module.weight_quantizer(module.temp_weight)
                module.stage_config = stage_cfg
            else:
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            module.use_temporary_parameter = True

def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, (QuantLinear, QuantMoE)):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            if not args.onlymlp:
                smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                        model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            if not args.onlymlp:
                smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                    model.out_smooth_scale, model.out_smooth_shift)
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
        if not args.onlymlp:
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
    # for name, module in model.named_modules():
    #     if isinstance(module, (QuantLinear, QuantMoE)):
    #         module.weight = module.weight_quantizer(module.weight)
    #         module.use_temporary_parameter=False

    if (args.slice_bits_list is not None) and  (args.target_bit_eval != 2.0):
        stage_cfg = {
            "stage_type": "buildq_routed",
            "slice_bit_list": args.slice_bits_list,
        }
    elif (args.slice_bits_list is not None) and (args.target_bit_eval == 2.0):
        stage_cfg = {
            'stage_type': 'msb_only',
            'bits': args.slice_bits_list[0],
        }

    for name, module in model.named_modules():
        if isinstance(module, (QuantMoE, QuantLinear)):
            if isinstance(module, QuantMoE):
                module.temp_weight = module.weight
                module.temp_bias = module.bias

                _ = module.weight_quantizer(module.temp_weight)
                module.stage_config = stage_cfg

                # ! Ensure router is in eval mode
                # ! - No need to freeze parameters
                module.Mask_Router.eval()
                # for p in module.Mask_Router.parameters():
                #     p.requires_grad = False
                module.use_temporary_parameter = True

            else:  # QuantLinear
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter = False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul, QuantMoE)):
            m.set_quant_state(weight_quant, act_quant)


def set_inference_mode(self, bitmoe_infer: bool = False, soft_infer: bool = False,hard_infer: bool = False, buildq_infer: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.bitmoe_infer = bitmoe_infer
    self.soft_infer = soft_infer
    self.hard_infer = hard_infer # soft hard is for bitmoe itself
    self.buildq_infer = buildq_infer
    for m in self.modules():
        if isinstance(m, QuantMoE):
            m.set_inference_mode(bitmoe_infer, buildq_infer)

def compute_stage_config(args):
    """Generate stage configurations from slice_bit_list."""
    slice_bit_list = args.slice_bits_list
    stage_weight = args.stage_weight
    assert stage_weight is not None, "Please provide stage weights for each stage."
    msb_bits = slice_bit_list[0]
    residual_bits = sum(slice_bit_list) - msb_bits

    stages = []
    # Stage 1: MSB only
    stages.append({
        'stage_type': 'msb_only',
        'bits': msb_bits,
        'weight': stage_weight[0]
    })

    # Stage 2: MSB + Residual
    # stages.append({
    #     'stage_type': 'msb_residual',
    #     'msb_bits': msb_bits,
    #     'residual_bits': residual_bits,
    #     'weight': stage_weight[0]
    # })

    # Stage 3: BuildQ routed
    stages.append({
        'stage_type': 'buildq_routed',
        'slice_bit_list': slice_bit_list,
        'weight': stage_weight[1]
    })

    return stages

def router_state_dict(model, destination=None, prefix='', keep_vars=False):
    """Extract only router parameters from model."""
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if 'router' in name:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def collect_router_stats(qlayer, quant_inps, attention_mask, position_ids,
                         position_embeddings, args, epoch, layer_idx):
    """
    Collect comprehensive router statistics for current layer and epoch.

    Args:
        qlayer: Quantized decoder layer containing QuantMoE modules
        quant_inps: Input tensors for the layer
        attention_mask: Attention mask for forward pass
        position_ids: Position IDs for forward pass
        position_embeddings: Position embeddings for forward pass
        args: Training arguments containing slice_bits_list, nsamples, batch_size
        epoch: Current training epoch
        layer_idx: Current layer index

    Returns:
        Dictionary containing:
            - summary: Layer-level aggregate statistics
            - modules: Per-module (gate_proj, up_proj, down_proj) statistics
            - token_sample: Sample of token-level routing decisions
    """
    stats = {
        'summary': {},
        'modules': {},
        'token_sample': {}
    }

    # Determine current training step for temperature calculation
    current_step = epoch * (args.nsamples // args.batch_size)

    # Forward pass to populate router.scores and router.scalar
    # Use a sample of tokens to avoid excessive computation
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda"):
            sample_size = min(1000, args.nsamples)
            for j in range(sample_size):
                _ = qlayer(quant_inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        step=current_step)

    # Collect from each QuantMoE module
    all_scores = []
    all_scalars = []
    all_sparsity = []
    module_names = []

    for name, module in qlayer.named_modules():
        if isinstance(module, QuantMoE):
            if hasattr(module.Mask_Router, 'scores') and hasattr(module.Mask_Router, 'scalar'):
                scores = module.Mask_Router.scores.detach().cpu()
                scalars = module.Mask_Router.scalar.detach().cpu()

                # Compute sparsity
                sparsity = module.get_sparsity()
                all_sparsity.append(sparsity)

                # Per-module stats
                stats['modules'][name] = {
                    'scores_mean': scores.mean(dim=0).tolist(),
                    'scores_std': scores.std(dim=0).tolist(),
                    'scalars_mean': scalars.mean(dim=0).tolist(),
                    'scalars_std': scalars.std(dim=0).tolist(),
                    'expert_activation_rates': (scalars > 0.5).float().mean(dim=0).tolist(),
                    'sparsity': sparsity
                }

                all_scores.append(scores)
                all_scalars.append(scalars)
                module_names.append(name)

    # Aggregate summary stats across all modules
    if all_scalars:
        combined_scalars = torch.cat(all_scalars, dim=0)
        base_bits = args.slice_bits_list[0]
        expert_bits = torch.tensor(args.slice_bits_list[1:], dtype=combined_scalars.dtype)
        effective_bits = base_bits + (combined_scalars * expert_bits).sum(dim=1)

        stats['summary'] = {
            'avg_effective_bits': effective_bits.mean().item(),
            'std_effective_bits': effective_bits.std().item(),
            'min_effective_bits': effective_bits.min().item(),
            'max_effective_bits': effective_bits.max().item(),
            'expert_activation_rates': combined_scalars.mean(dim=0).tolist(),
            'temperature': _compute_temperature(current_step, args.global_steps),
            'avg_sparsity': sum(all_sparsity) / len(all_sparsity) if all_sparsity else 0.0
        }

        # Save token sample for detailed analysis (from first module only to save space)
        sample_idx = 0
        stats['token_sample'] = {
            'scores': all_scores[sample_idx][:1000].clone(),
            'scalars': all_scalars[sample_idx][:1000].clone(),
            'effective_bits': effective_bits[:1000].clone()
        }

    return stats

def _compute_temperature(step, global_steps):
    """Compute temperature value for router annealing (matches GateMask.step())."""
    max_temperature = 100

    if step < global_steps:
        temperature = min(max_temperature,
                         1.0 / (1 - torch.log(torch.tensor(max(1, step), dtype=torch.float32)) /
                                torch.log(torch.tensor(global_steps, dtype=torch.float32)) + 1e-6))
    else:
        temperature = max_temperature

    return temperature.item() if isinstance(temperature, torch.Tensor) else temperature