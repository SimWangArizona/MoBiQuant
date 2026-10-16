import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer,TruncPTQQuantizer
import math
from parser import args

def hard_gate(x, threshold=0.5):
    return (x >= threshold).float().to(x)


class QuantMoE(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        bit_list: list = [],
        global_steps: int = 1000,
        disable_input_quant=False,
        use_shared_expert=True,
    ):
        super().__init__()
        self.bit_list = bit_list
        self.num_experts = len(bit_list)
        self.global_steps = global_steps
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.use_shared_expert = use_shared_expert


        self.Mask_Router = GateMask(n_experts=self.num_experts - 1, global_step=self.global_steps, input_size=self.in_features) 
        self.weight_quantizer = TruncPTQQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        
        
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.bit_moe_infer = False
        self.buildq_infer = False

        self.stage_config = None    # 'msb_only', 'msb_residual', 'buildq_routed'
        
    def get_sparsity(self):
        with torch.no_grad():
            if self.training: 
                binary_mask = hard_gate(self.Mask_Router.scalar, threshold=0.5)
            else:
                threshold = self.Mask_Router._compute_eval_threshold()
                binary_mask = hard_gate(self.Mask_Router.scores, threshold=threshold)
            total_elements = binary_mask.numel()
            zero_elements = (binary_mask == 0).sum().item()
            sparsity = zero_elements / total_elements
        return sparsity

    def get_effective_bits(self):
        """Calculate average effective bits based on router activations.

        Formula: avg_effective_bits = base_bits + mean(hard_gate(scores) * expert_bits)
        where hard_gate applies the same threshold used during inference.
        """
        with torch.no_grad():
            if len(self.bit_list) <= 1:
                # No routing, only base bits
                return float(self.bit_list[0]) if self.bit_list else 0.0

            base_bits = float(self.bit_list[0])
            expert_bits = torch.tensor(self.bit_list[1:], dtype=self.Mask_Router.scalar.dtype, device=self.Mask_Router.scalar.device)

            # Use hard gates (binary 0/1) with dynamic threshold to match inference behavior
            if self.training:
                hard_gates = hard_gate(self.Mask_Router.scalar, threshold=0.5)
            else:
                threshold = self.Mask_Router._compute_eval_threshold()
                hard_gates = hard_gate(self.Mask_Router.scores, threshold=threshold)

            # scalar shape: (num_tokens, num_experts-1)
            # expert_bits shape: (num_experts-1,)
            # Compute per-token effective bits, then average
            effective_bits_per_token = base_bits + (hard_gates * expert_bits).sum(dim=1)
            avg_effective_bits = effective_bits_per_token.mean().item()

            return avg_effective_bits
    def get_token_bit_distribution(self):
        """
        Compute per-token effective bits and return a distribution.
        Returns:
            dict: {bit_value: percentage_of_tokens}
        """
        with torch.no_grad():
            if len(self.bit_list) <= 1:
                # All tokens use only base bits
                bit_val = float(self.bit_list[0]) if self.bit_list else 0.0
                return {bit_val: 100.0}

            base_bits = float(self.bit_list[0])
            expert_bits = torch.tensor(
                self.bit_list[1:], dtype=self.Mask_Router.scalar.dtype, device=self.Mask_Router.scalar.device
            )

            # 获取 hard gates
            if self.training:
                hard_gates = hard_gate(self.Mask_Router.scalar, threshold=0.5)
            else:
                threshold = self.Mask_Router._compute_eval_threshold()
                hard_gates = hard_gate(self.Mask_Router.scores, threshold=threshold)
            
            # 每个 token 实际使用的 bits
            # hard_gates: (num_tokens, num_experts-1)
            # expert_bits: (num_experts-1,)
            token_bits = base_bits + (hard_gates * expert_bits).sum(dim=1)  # shape: (num_tokens,)

            # 转成整数 bit 值（可根据需求保留小数）
            token_bits_int = token_bits.round().int().tolist()

            # 统计每个 bit 值的 token 数量
            from collections import Counter

            count = Counter(token_bits_int)
            total_tokens = len(token_bits_int)

            # 转换为百分比
            distribution = {bit: 100.0 * cnt / total_tokens for bit, cnt in count.items()}

            # 返回按 bit 排序的字典
            return dict(sorted(distribution.items()))

    def forward(self, input: torch.Tensor,step=None):
        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input)

        if self.use_temporary_parameter:
            if self.stage_config['stage_type'] == 'msb_only':
                weight, bias = self._forward_msb_only()
                out = self.fwd_func(input, weight, bias)
            elif self.stage_config['stage_type'] == 'msb_residual':
                weight, bias = self._forward_msb_residual()
                out = self.fwd_func(input, weight, bias)
            elif self.stage_config['stage_type'] == 'buildq_routed':
                out = self._forward_buildq_routed(input, step)
            else: 
                weight, bias = self.temp_weight, self.temp_bias
                out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        else:   # % Fallback for old training code (non-BuildQ temporary parameter mode)
            weight, bias = self.weight, self.bias
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    def set_inference_mode(self, bitmoe_infer: bool = False, buildq_infer: bool = False):
        self.bit_moe_infer = bitmoe_infer
        self.buildq_infer = buildq_infer

    def _reshape_for_groupwise(self, x):
        """Reshape tensor for group-wise quantization if needed.
        Args:
            x: Input tensor with shape [out_features, in_features]
        Returns:
            tuple: (reshaped_tensor, original_shape) where reshaped_tensor has shape
                [num_groups, group_size] and original_shape is None if no reshaping occurred
        """
        group_size = self.weight_quantizer.group_size
        deficiency = self.weight_quantizer.deficiency
        original_shape = x.shape

        # Add padding if needed (matches quantizer.py lines 246-248)
        if deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0], deficiency), dtype=x.dtype, device=x.device)
            x = torch.cat((x, pad_zeros), dim=1)

        # Reshape to groups (matches quantizer.py lines 250-253)
        x = x.reshape(-1, group_size)

        return x, original_shape

    def _reshape_from_groupwise(self, x, original_shape):
        deficiency = self.weight_quantizer.deficiency
        # Reshape back to original dimensions (matches quantizer.py lines 269-270)
        x = x.reshape(original_shape[0], -1)
        # Remove padding if needed (matches quantizer.py lines 271-272)
        if deficiency > 0:
            x = x[:, :-deficiency]

        return x

    def _forward_msb_only(self):
        """Stage 1: Extract MSB k bits only."""
        tmp_w = self.temp_weight
        weight = self.weight_quantizer(tmp_w)
        
        with torch.no_grad():
            scale = self.weight_quantizer.scale  # Scale for msb_bits
            zero = self.weight_quantizer.round_zero_point  # Zero for msb_bits
            msb_bits = self.stage_config['bits']  # 4
            qmax = (2 ** msb_bits) - 1  # 15 for 4 bits

            # Reshape for group-wise quantization if needed
            if self.weight_quantizer.group_size:
                tmp_w_reshaped, original_shape = self._reshape_for_groupwise(tmp_w)
                msb_int = torch.clamp(torch.floor(tmp_w_reshaped / scale + zero), 0, qmax).to(torch.uint8)
                msb_int = self._reshape_from_groupwise(msb_int, original_shape)
            else:
                msb_int = torch.clamp(torch.floor(tmp_w / scale + zero), 0, qmax).to(torch.uint8)
            self.weight_quantizer.MSBk_int = msb_int
        
        return weight, self.temp_bias

    def _forward_msb_residual(self):
        """Stage 2: MSB k bits + residual bits quantization."""
        tmp_w = self.temp_weight
        scale_k = self.weight_quantizer.scale  # Scale for msb_bits
        zero_k = self.weight_quantizer.round_zero_point  # Zero for msb_bits
        msb_bits = self.stage_config['msb_bits']  # 4
        residual_bits = self.stage_config['residual_bits']  # 4
        
        # Step 1: MSB k-bit quantization
        tmp_wq_k = self.weight_quantizer(tmp_w)
        
        # Step 2: Residual quantization
        tmp_r = tmp_w - tmp_wq_k
        qmax_r = (2 ** residual_bits) - 1  # 15
        scale_r = scale_k / (2 ** residual_bits)  # Finer scale for residual
        zero_r = (qmax_r + 1) / 2.0  # 8.0 (centered)
        
        # Reshape for group-wise quantization if needed
        if self.weight_quantizer.group_size:
            tmp_r_reshaped, original_shape = self._reshape_for_groupwise(tmp_r)
            lsb_int = torch.clamp(torch.floor(tmp_r_reshaped / scale_r + zero_r), 0, qmax_r)
            tmp_wq_r_reshaped = scale_r * (lsb_int - zero_r + 0.5)
            tmp_wq_r = self._reshape_from_groupwise(tmp_wq_r_reshaped, original_shape)
            lsb_int_reshaped = self._reshape_from_groupwise(lsb_int, original_shape)
            self.weight_quantizer.LSBr_int = lsb_int_reshaped.to(torch.uint8)
        else:
            lsb_int = torch.clamp(torch.floor(tmp_r / scale_r + zero_r), 0, qmax_r)
            tmp_wq_r = scale_r * (lsb_int - zero_r + 0.5)
            self.weight_quantizer.LSBr_int = lsb_int.to(torch.uint8)
        
        # Step 3: Combine MSB + residual
        weight = tmp_wq_k + tmp_wq_r
        
        return weight, self.temp_bias

    def _forward_buildq_routed(self, input, step):
        """Stage 3: Multi-level residual quantization with routing."""
        bsz, seqlen, d_in = input.shape
        tmp_w = self.temp_weight
        scale = self.weight_quantizer.scale  # Base scale for MSB bits
        zero = self.weight_quantizer.round_zero_point  # Base zero for MSB bits
        
        slice_bit_list = self.stage_config['slice_bit_list']  # e.g., [4, 2, 2]
        experts_weight = {}
        
        # Multi-level residual quantization
        if self.weight_quantizer.group_size:
            # Reshape once at the start
            tmp_w_reshaped, original_shape = self._reshape_for_groupwise(tmp_w)
            residual = tmp_w_reshaped
            current_scale = scale

            for idx, bit_width in enumerate(slice_bit_list):
                qmax = (2 ** bit_width) - 1

                if idx == 0:
                    # Expert 0 (MSB): Use original scale and zero
                    w_int = torch.clamp(torch.floor(residual / current_scale + zero), 0, qmax)
                    experts_weight_reshaped = current_scale * (w_int - zero + 0.5)
                else:
                    # Experts 1-n (residual): Use centered zero-point
                    zero_residual = (qmax + 1) / 2.0
                    w_int = torch.clamp(torch.floor(residual / current_scale + zero_residual), 0, qmax)
                    experts_weight_reshaped = current_scale * (w_int - zero_residual + 0.5)

                # Reshape back to original dimensions for forward pass
                experts_weight[idx] = self._reshape_from_groupwise(experts_weight_reshaped, original_shape)

                # Update residual for next expert (in reshaped space)
                residual = residual - experts_weight_reshaped

                # Scale becomes finer for next expert
                current_scale = current_scale / (2 ** bit_width)
        else:
            # No group-wise quantization, use original code
            residual = tmp_w
            current_scale = scale

            for idx, bit_width in enumerate(slice_bit_list):
                qmax = (2 ** bit_width) - 1

                if idx == 0:
                    w_int = torch.clamp(torch.floor(residual / current_scale + zero), 0, qmax)
                    experts_weight[idx] = current_scale * (w_int - zero + 0.5)
                else:
                    zero_residual = (qmax + 1) / 2.0
                    w_int = torch.clamp(torch.floor(residual / current_scale + zero_residual), 0, qmax)
                    experts_weight[idx] = current_scale * (w_int - zero_residual + 0.5)

                residual = residual - experts_weight[idx]
                current_scale = current_scale / (2 ** bit_width)

        
        # --- Minimal device/dtype fix begins here ---
        for k in list(experts_weight.keys()):
            experts_weight[k] = experts_weight[k].to(device=input.device, dtype=input.dtype)
        
        # Forward pass with routing
        # Expert 0 is always shared (no routing)
        out = self.fwd_func(input.reshape(bsz * seqlen, -1), experts_weight[0], self.temp_bias, **self.fwd_kwargs)
        
        # Experts 1-n are routed
        if len(slice_bit_list) > 1:
            collected_inputs = self.Mask_Router(input, step=step)  # [bsz*seqlen, num_experts-1, hidden]
            for expert_idx in range(1, len(slice_bit_list)):
                expert_output = self.fwd_func(
                    collected_inputs[:, expert_idx - 1, :],
                    experts_weight[expert_idx],
                    self.temp_bias,
                    **self.fwd_kwargs
                )
                out += expert_output
        
        out = out.reshape(bsz, seqlen, -1)
        return out


class GateMask(nn.Module):
    def __init__(self, n_experts, input_size, global_step=10):
        super(GateMask, self).__init__()
        self.global_step = global_step
        self.global_threshold = None  # Cached threshold from calibration

        self.router = nn.Sequential(
            nn.Linear(input_size, 2048, bias=False),    # (n,2048), n = (bsz)x(seqlen)
            nn.Linear(2048, n_experts, bias=False)      # (2048,e), e = n_experts
        )
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.router[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.router[1].weight, a=math.sqrt(5))

    def _compute_eval_threshold(self):
        """Compute the threshold for hard gating based on training/eval mode.
        During eval: Returns flexible threshold based on args (target_activation_ratio_eval or target_bit_eval)

        This ensures statistics collection matches the actual behavior in each mode.
        """
        # If a global threshold has been calibrated, use it (fastest path)
        if self.global_threshold is not None:
            return self.global_threshold

        # Otherwise, compute threshold dynamically from current scores
        router_scores = self.scores.detach()  # Use scores instead of scalar to avoid saturation
        # Eval: Use flexible threshold based on args
        if args.target_activation_ratio_eval is not None:
            routing_ratio = args.target_activation_ratio_eval
            threshold = torch.quantile(router_scores.float(), 1 - routing_ratio)
        elif args.target_bit_eval is not None:
            msb_bit = args.slice_bits_list[0]
            residual_bits_sum = sum(args.slice_bits_list[1:])
            routing_ratio = (args.target_bit_eval - msb_bit) / residual_bits_sum
            threshold = torch.quantile(router_scores.float(), 1 - routing_ratio)
        else:
            threshold = 0.0  # Default threshold for scores (instead of 0.5 for scalar)
        return threshold

    def forward(self, x, step):
        x = x.reshape(-1, x.size(-1))   # (n,d)
        self.scores = self.router(x)    # (n,e)
        self.scalar = self.step(step)   # (n,e) - soft values [0,1]
        # self.scalar = self.step(step=args.global_steps*0.9)   # (n,e) - soft values [0,1]

        # Apply hard gating in eval mode
        if self.training:
            # Training: Use soft gates for gradient flow
            scalar_forward = self.scalar
        else:
            # Eval: Use hard binary gates based on scores (pre-sigmoid) to avoid saturation
            threshold = self._compute_eval_threshold()
            # Threshold on scores, not scalar, to avoid tie-breaking issues from saturation
            scalar_forward = hard_gate(self.scores, threshold=threshold)

        # (n,e,d) = (n,1,d) * (n,e,1)
        output = x.unsqueeze(1) * scalar_forward.unsqueeze(2).to(x)
        return output
    
    def step(self, step=1):
        if step is None:
            step = self.global_step
        max_temperature = 100
        
        if step < self.global_step:
            ratio = math.log(max(1, int(step))) / math.log(self.global_step)    # maps [1,G] -> [0,1]
            temperature = min(max_temperature, 1.0 / (1.0 - ratio + 1e-6))      # maps [0,1] -> [1,max_temperature]
        else:
            temperature = max_temperature
        
        mask_in = self.scores * temperature             # Scale router scores to sharpen gating
        scalar = torch.sigmoid(mask_in)                 # Obtain gating values in [0,1]

        return scalar
    
    def __repr__(self):
        return f"{self.__class__.__name__}(trainable_mask={self.trainable_mask}, global_step={self.global_step}, alpha={self.alpha}, hidden_size={self.scalar.size(-1)}, dimdown_dim={self.dimdown_dim})"
