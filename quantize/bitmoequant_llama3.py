import torch
import torch.nn as nn
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_moe import QuantMoE
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import copy
import utils
import os
import pdb
import gc
# import wandb
from quantize.utils import find_grad_norm, let_parameters, lwc_parameters,bitmoe_parameters, get_omni_parameters,get_bitmoe_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state,set_inference_mode,\
                            compute_stage_config, router_state_dict, collect_router_stats
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def bitmoequant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        if args.onlymlp:
            from models.int_llama_layer_onlyMLP import QuantLlamaDecoderLayer
            DecoderLayer = QuantLlamaDecoderLayer
            print("Only MLP quantization is activated.")
            pairs = {
                "up_proj":"fc1"
            }
        else:
            # from models.int_llama3_layer_whole_moe import QuantLlamaDecoderLayer
            from models.int_llama_layer import QuantLlamaDecoderLayer
            # from models.int_llama_layer_LWC_MOE import QuantLlamaDecoderLayer
            DecoderLayer = QuantLlamaDecoderLayer
            pairs = {
                "q_proj":"qkv",
                "o_proj":"out",
                "up_proj":"fc1"
            }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float16
        traincast = torch.amp.autocast
    else:
        dtype = torch.float16
        traincast = torch.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.rotary_emb = LlamaRotaryEmbedding(config=model.config)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                # print("Using Llama model when obtain position_ids")

                cache["position_ids"] = kwargs["position_ids"]
                cache["position_embeddings"] = self.rotary_emb(inp, kwargs["position_ids"])

            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]


    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None


    # load omni parameters if resume from a checkpoint
    if args.resume:
        if os.path.isdir(args.resume):
            omni_parameters = torch.load(os.path.join(args.resume, "omni_parameters.pth"))
            router_parameters = torch.load(os.path.join(args.resume, "router_parameters.pth"))
            router_stats = torch.load(os.path.join(args.resume, "router_stats.pth"))
            args.global_steps = router_stats['config']['global_steps']  # NEW
        elif os.path.isfile(args.resume):
            omni_parameters = torch.load(args.resume)
        else:
            raise ValueError("args.resume should be a dir or a file")
    else:
        omni_parameters = {}
        router_parameters = {}  # NEW
        args.global_steps = (args.nsamples // args.batch_size) * args.epochs
        # Initialize router statistics collection
        router_stats = {
            'config': {
                'slice_bit_list': args.slice_bits_list if args.slice_bits_list else [],
                'n_layers': len(layers),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'nsamples': args.nsamples,
                'global_steps': args.global_steps
            },
            'layers': {}
        }

    # Initialize wandb step counter for linear W&B logging
    wandb_step = 0
    wandb_total_steps = len(layers) * args.epochs if args.epochs > 0 else 0

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():
            raise ValueError("Mixtral architecture is not supported yet. Only LLaMA is supported for now.")

        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        #region Obtain FP outputs for each layer 
        # set_inference_mode(qlayer, bitmoe_infer=False, buildq_infer=False) # ensure use original forward
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):
                    for j in range(args.nsamples):
                        # input fpinps to full precision model to get output, replace fpinput with current layer's output
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids,position_embeddings=position_embeddings)
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids,position_embeddings=position_embeddings)
        #endregion

        #region Activation quant. and LET
        # * init smooth parameters
        # * skip this because we only do weight quant. for now
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, (QuantLinear, QuantMoE)):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
        #endregion

        if args.resume:
            if os.path.isdir(args.resume):
                qlayer.load_state_dict(omni_parameters[i], strict=False)
                qlayer.load_state_dict(router_parameters[i], strict=False)  # NEW
            elif os.path.isfile(args.resume):
                qlayer.load_state_dict(omni_parameters[i], strict=False)
            
        if args.epochs > 0:
            
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            
            # % Optimzer & Regularization Parameters
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, 
                 {"params":lwc_parameters(qlayer),"lr":args.lwc_lr},
                 {"params":bitmoe_parameters(qlayer),"lr":args.bitmoe_lr}], weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            stage_configs = compute_stage_config(args) if args.slice_bits_list is not None else None
            # lamda_r = [0.1,1.0]

            for epochs in range(args.epochs):
                loss_list, norm_list = [], []
                sparsity_list, eff_bits_list = [], []
                
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    optimizer.zero_grad()
                    total_loss = 0.0

                    if args.slice_bits_list is not None:
                        assert args.wbits == args.slice_bits_list[0], "msb_bit should be slice_bits_list[0]"
                        # # Stage-aware quantization
                        with traincast(device_type="cuda"):
                            for stage_idx, stage_cfg in enumerate(stage_configs):
                                # Freeze LWC parameters for Stage 3
                                if stage_cfg['stage_type'] == 'buildq_routed':
                                    # for param in lwc_parameters(qlayer):
                                    #     param.requires_grad = False
                                        current_step = epochs * (args.nsamples // args.batch_size) + j + 1
                                else:
                                    # for param in lwc_parameters(qlayer):
                                    #     param.requires_grad = True
                                        current_step = None
                                smooth_and_quant_temporary(qlayer, args, is_llama, buildq_mode=True, stage_cfg=stage_cfg)
                                quant_out = qlayer(
                                    quant_inps[index:index+args.batch_size,],
                                    attention_mask=attention_mask_batch,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings,
                                    step=current_step)
                                loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                                if args.aug_loss:
                                    loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                                loss *= stage_cfg['weight']
                                # logger.info(f"Layer {i} Epoch {epochs} Stage {stage_idx} Loss: {loss.item()}")
                                total_loss += loss
                                # only calculate reg loss for MoE router
                                if stage_cfg['stage_type'] == 'buildq_routed':
                                    # Compute layer-wise regularization loss
                                    reg_loss, module_stats = qlayer.get_layer_reg_loss(args, step=current_step)
                                    reg_loss *= args.reg_weight
                                    total_loss += reg_loss
                                    # Store average sparsity and effective bits
                                    module_sparsity = module_stats['sparsity']
                                    module_eff_bits = module_stats['eff_bits']
                                    sparsity_list.append(sum(module_sparsity.values()) / len(module_sparsity))
                                    eff_bits_list.append(sum(module_eff_bits.values()) / len(module_eff_bits))
                                    
                    else:
                        # Standard OmniQuant quantization
                        with traincast(device_type="cuda"):
                            smooth_and_quant_temporary(qlayer, args, is_llama, buildq_mode=False, stage_cfg=None)
                            quant_out = qlayer(
                                quant_inps[index:index+args.batch_size,],
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)
                            loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                            if args.aug_loss:
                                loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                            total_loss = loss
                    
                    loss_list.append(total_loss.detach().cpu())
                    # ! add clip grad for bitmoe parameters
                    norm = loss_scaler(total_loss, optimizer, parameters=get_bitmoe_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                sparsity_mean = sum(sparsity_list) / len(sparsity_list) if sparsity_list else 0.0
                eff_bits_mean = sum(eff_bits_list) / len(eff_bits_list) if eff_bits_list else 0.0
                mem_allocated = torch.cuda.max_memory_allocated(lm._device) / 1024**2
                logger.info(
                    f"layer {i} iter {epochs} loss:{loss_mean:.6f} norm:{norm_mean:.6f} sparsity:{sparsity_mean:.4f} eff_bits:{eff_bits_mean:.3f} max memory_allocated {mem_allocated:.4f}"
                )

                # Log to W&B with per-layer metrics (same graph, different colors per layer)
                if args.wandb:
                    log_dict = {
                        "epoch": epochs,
                        "progress": wandb_step / wandb_total_steps if wandb_total_steps > 0 else 0,
                    }
                    # Only log loss, sparsity, and effective_bits when i is divisible by 2
                    if i % 4 == 0:
                        log_dict[f"loss/layer_{i}"] = loss_mean.item()
                        log_dict[f"sparsity/layer_{i}"] = sparsity_mean
                        log_dict[f"effective_bits/layer_{i}"] = eff_bits_mean
                    wandb.log(log_dict, step=wandb_step)
                    wandb_step += 1
                # # Collect router statistics (only if using BuildQ with routing)
                # if args.slice_bits_list is not None and len(args.slice_bits_list) > 1:
                #     epoch_stats = collect_router_stats(
                #         qlayer, quant_inps, attention_mask, position_ids,
                #         position_embeddings, args, epochs, i
                #     )
                #     # Store in hierarchy
                #     if i not in router_stats['layers']:
                #         router_stats['layers'][i] = {'epochs': {}}
                #     router_stats['layers'][i]['epochs'][epochs] = epoch_stats
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half()

        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast(device_type="cuda"):
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(
                            quant_inps[j].unsqueeze(0), 
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings)
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
            router_parameters[i] = router_state_dict(qlayer)  # NEW
            torch.save(router_parameters, os.path.join(args.output_dir, f"router_parameters.pth"))  # NEW
            # Save router statistics incrementally
            if args.slice_bits_list is not None and len(args.slice_bits_list) > 1:
                torch.save(router_stats, os.path.join(args.output_dir, "router_stats.pth"))
                logger.info(f"Saved router statistics for layer {i} to router_stats.pth")
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
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
