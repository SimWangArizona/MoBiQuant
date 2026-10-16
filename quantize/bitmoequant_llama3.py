import torch
import torch.nn as nn
from quantize.int_moe import QuantMoE
from quantize.int_linear import QuantLinear
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import copy
import utils
import os
import gc
from quantize.checkpoint import load_moe_checkpoint
from quantize.utils import let_parameters, lwc_parameters, bitmoe_parameters, get_bitmoe_parameters, omni_state_dict, register_scales_and_zeros, smooth_and_quant_temporary, smooth_and_quant_inplace, clear_temp_variable, set_quant_state, compute_stage_config, router_state_dict

def bitmoequant(lm, args, dataloader, act_scales, act_shifts, logger=None):
    logger.info('Starting ...')
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = True
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    from models.int_llama_layer import QuantLlamaDecoderLayer
    DecoderLayer = QuantLlamaDecoderLayer
    if args.slice_bits_list is not None:
        logger.info('Attention q/k/v/o projections: BitMoE, base bits=%s, residual slices=%s; effective bits depend on routing, target_bit_eval=%s', args.slice_bits_list[0], args.slice_bits_list[1:], args.target_bit_eval)
    else:
        logger.info('Attention q/k/v/o projections: fixed weight bits=%s', args.wbits)
    pairs = {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}
    layer_name_prefix = 'model.layers'
    layers[0] = layers[0].to(dev)
    dtype = torch.float16
    traincast = torch.amp.autocast
    inps = torch.zeros((args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.rotary_emb = LlamaRotaryEmbedding(config=model.config)

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if self.is_llama:
                cache['position_ids'] = kwargs['position_ids']
                cache['position_embeddings'] = self.rotary_emb(inp, kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    with torch.no_grad():
        for batch in dataloader:
            if cache['i'] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings']
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1) if args.deactive_amp else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logger.info("No attention mask caught from the first layer. Seems that model's attention works without a mask.")
        attention_mask_batch = None
    loss_func = torch.nn.MSELoss()
    position_ids = cache['position_ids']
    if args.resume:
        if os.path.isdir(args.resume):
            omni_parameters, router_parameters, router_stats, args.global_steps = load_moe_checkpoint(args.resume, args.epochs, logger)
        elif os.path.isfile(args.resume):
            omni_parameters = torch.load(args.resume, map_location='cpu')
        else:
            raise ValueError('args.resume should be a dir or a file')
    else:
        omni_parameters = {}
        router_parameters = {}
        args.global_steps = args.nsamples // args.batch_size * args.epochs
        router_stats = {'config': {'slice_bit_list': args.slice_bits_list if args.slice_bits_list else [], 'n_layers': len(layers), 'epochs': args.epochs, 'batch_size': args.batch_size, 'nsamples': args.nsamples, 'global_steps': args.global_steps}, 'layers': {}}
    for i in range(len(layers)):
        logger.info(f'=== Start quantize layer {i} ===')
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        qlayer.let = args.let
        use_shift = True
        use_shift = False
        if args.let:
            qlayer.register_parameter('qkt_smooth_scale', torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features, device=dev, dtype=dtype)))
            for name, module in qlayer.named_modules():
                if isinstance(module, (QuantLinear, QuantMoE)):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f'{layer_name_prefix}.{i}.{name}'].to(device=dev, dtype=dtype).clamp(min=1e-05)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-05)
                            scale = (act.pow(args.alpha) / weight.pow(1 - args.alpha)).clamp(min=1e-05)
                            shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f'{pairs[key]}_smooth_shift', torch.nn.Parameter(shift))
                            qlayer.register_parameter(f'{pairs[key]}_smooth_scale', torch.nn.Parameter(scale))
        if args.resume:
            if os.path.isdir(args.resume):
                qlayer.load_state_dict(omni_parameters[i], strict=False)
                qlayer.load_state_dict(router_parameters[i], strict=False)
            elif os.path.isfile(args.resume):
                qlayer.load_state_dict(omni_parameters[i], strict=False)
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()
            optimizer = torch.optim.AdamW([{'params': let_parameters(qlayer, use_shift), 'lr': args.let_lr}, {'params': lwc_parameters(qlayer), 'lr': args.lwc_lr}, {'params': bitmoe_parameters(qlayer), 'lr': args.bitmoe_lr}], weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            stage_configs = compute_stage_config(args) if args.slice_bits_list is not None else None
            for epochs in range(args.epochs):
                loss_list, norm_list = ([], [])
                sparsity_list, eff_bits_list = ([], [])
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    optimizer.zero_grad()
                    total_loss = 0.0
                    if args.slice_bits_list is not None:
                        assert args.wbits == args.slice_bits_list[0], 'msb_bit should be slice_bits_list[0]'
                        with traincast(device_type='cuda'):
                            for stage_idx, stage_cfg in enumerate(stage_configs):
                                if stage_cfg['stage_type'] == 'buildq_routed':
                                    current_step = epochs * (args.nsamples // args.batch_size) + j + 1
                                else:
                                    current_step = None
                                smooth_and_quant_temporary(qlayer, args, is_llama, buildq_mode=True, stage_cfg=stage_cfg)
                                quant_out = qlayer(quant_inps[index:index + args.batch_size,], attention_mask=attention_mask_batch, position_ids=position_ids, position_embeddings=position_embeddings, step=current_step)
                                loss = loss_func(fp_inps[index:index + args.batch_size,], quant_out)
                                if args.aug_loss:
                                    loss += loss_func(fp_inps_2[index:index + args.batch_size,], quant_out)
                                loss *= stage_cfg['weight']
                                total_loss += loss
                                if stage_cfg['stage_type'] == 'buildq_routed':
                                    reg_loss, module_stats = qlayer.get_layer_reg_loss(args, step=current_step)
                                    reg_loss *= args.reg_weight
                                    total_loss += reg_loss
                                    module_sparsity = module_stats['sparsity']
                                    module_eff_bits = module_stats['eff_bits']
                                    sparsity_list.append(sum(module_sparsity.values()) / len(module_sparsity))
                                    eff_bits_list.append(sum(module_eff_bits.values()) / len(module_eff_bits))
                    else:
                        with traincast(device_type='cuda'):
                            smooth_and_quant_temporary(qlayer, args, is_llama, buildq_mode=False, stage_cfg=None)
                            quant_out = qlayer(quant_inps[index:index + args.batch_size,], attention_mask=attention_mask_batch, position_ids=position_ids, position_embeddings=position_embeddings)
                            loss = loss_func(fp_inps[index:index + args.batch_size,], quant_out)
                            if args.aug_loss:
                                loss += loss_func(fp_inps_2[index:index + args.batch_size,], quant_out)
                            total_loss = loss
                    loss_list.append(total_loss.detach().cpu())
                    norm = loss_scaler(total_loss, optimizer, parameters=get_bitmoe_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                sparsity_mean = sum(sparsity_list) / len(sparsity_list) if sparsity_list else 0.0
                eff_bits_mean = sum(eff_bits_list) / len(eff_bits_list) if eff_bits_list else 0.0
                mem_allocated = torch.cuda.max_memory_allocated(lm._device) / 1024 ** 2
                logger.info(f'layer {i} iter {epochs} loss:{loss_mean:.6f} norm:{norm_mean:.6f} sparsity:{sparsity_mean:.4f} eff_bits:{eff_bits_mean:.3f} max memory_allocated {mem_allocated:.4f}')
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half()
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs > 0:
            with torch.no_grad():
                with traincast(device_type='cuda'):
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to('cpu')
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f'omni_parameters.pth'))
            router_parameters[i] = router_state_dict(qlayer)
            torch.save(router_parameters, os.path.join(args.output_dir, f'router_parameters.pth'))
            if args.slice_bits_list is not None and len(args.slice_bits_list) > 1:
                torch.save(router_stats, os.path.join(args.output_dir, 'router_stats.pth'))
                logger.info(f'Saved router statistics for layer {i} to router_stats.pth')
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to('cpu')
        del layer
        torch.cuda.empty_cache()
    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache
    return model
