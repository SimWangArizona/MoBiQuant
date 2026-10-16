import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.bitmoequant_llama3 import bitmoequant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_moe import QuantMoE

import pdb
# import wandb

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def collect_moe_stats(lm, args, dataloader, logger):
    """Collect sparsity and effective bits from all QuantMoE modules after running forward passes.

    This function:
    1. Runs forward passes on calibration data to update router statistics
    2. Collects sparsity and effective bits from all QuantMoE modules
    3. Logs the statistics
    """
    logger.info("Collecting QuantMoE statistics from calibration data...")

    # Set model to eval mode
    lm.model.eval()

    # Prepare the model for forward passes
    dev = lm.device
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        layers = lm.model.model.layers
    elif "opt" in args.net.lower():
        layers = lm.model.model.decoder.layers
    elif "falcon" in args.net.lower():
        layers = lm.model.transformer.h
    else:
        raise ValueError("Unsupported model architecture")

    # Ensure all QuantMoE routers are on the correct device
    for name, module in lm.model.named_modules():
        if isinstance(module, QuantMoE):
            module.Mask_Router.to(dev)

    # Run forward passes on calibration data to update router statistics
    for i, batch in enumerate(dataloader):
        if i >= args.nsamples: # only across args.nsamples
            break
        _ = lm.model(batch[0].to(dev))

    # Collect statistics from all QuantMoE modules
    all_sparsity = []
    all_eff_bits = []
    all_token_bit_distributions = []
    layer_stats = {}  # Group by layer: {layer_idx: {module_type: {sparsity, eff_bits}}}

    for name, module in lm.model.named_modules():
        if isinstance(module, QuantMoE):
            sparsity = module.get_sparsity()
            eff_bits = module.get_effective_bits()
            token_bit_distribution = module.get_token_bit_distribution()
            all_sparsity.append(sparsity)
            all_eff_bits.append(eff_bits)
            all_token_bit_distributions.append(token_bit_distribution)
            # Parse layer index and module type from name
            # e.g., "model.layers.0.mlp.gate_proj" -> layer=0, module="gate_proj"
            parts = name.split('.')
            layer_idx = None
            module_type = None

            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                if i == len(parts) - 1:
                    module_type = part

            if layer_idx is not None and module_type is not None:
                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = {}
                layer_stats[layer_idx][module_type] = {'sparsity': sparsity, 'effective_bits': eff_bits, 'token_bit_distribution': token_bit_distribution}

    if all_sparsity:
        avg_sparsity = sum(all_sparsity) / len(all_sparsity)
        avg_eff_bits = sum(all_eff_bits) / len(all_eff_bits)

        logger.info("=" * 80)
        logger.info(f"QuantMoE Statistics (collected over {args.nsamples} samples):")
        logger.info(f"Total QuantMoE modules: {len(all_sparsity)}")
        logger.info(f"Average Sparsity: {avg_sparsity:.4f} ({avg_sparsity*100:.2f}%)")
        logger.info(f"Average Effective Bits: {avg_eff_bits:.3f}")
        logger.info("=" * 80)


        # Compute layer-wise statistics
        if layer_stats:
            logger.info("Layer-wise Statistics:")
            logger.info(f"{'Layer':<8} {'Q':<8} {'K':<8} {'V':<8} {'O':<8} {'Down':<8} {'Gate':<8} {'Up':<8} {'Mean Spar':<12} {'Mean Bits':<10}")
            logger.info("-" * 100)

            for layer_idx in sorted(layer_stats.keys()):
                # Get individual module stats for attention
                q_sp = layer_stats[layer_idx].get('q_proj', {}).get('sparsity', 0) * 100
                k_sp = layer_stats[layer_idx].get('k_proj', {}).get('sparsity', 0) * 100
                v_sp = layer_stats[layer_idx].get('v_proj', {}).get('sparsity', 0) * 100
                o_sp = layer_stats[layer_idx].get('o_proj', {}).get('sparsity', 0) * 100

                # Get individual module stats for MLP
                down_sp = layer_stats[layer_idx].get('down_proj', {}).get('sparsity', 0) * 100
                gate_sp = layer_stats[layer_idx].get('gate_proj', {}).get('sparsity', 0) * 100
                up_sp = layer_stats[layer_idx].get('up_proj', {}).get('sparsity', 0) * 100

                # Compute mean across all modules in this layer
                layer_sparsity = [stats['sparsity'] for stats in layer_stats[layer_idx].values()]
                layer_eff_bits = [stats['effective_bits'] for stats in layer_stats[layer_idx].values()]

                mean_sparsity = sum(layer_sparsity) / len(layer_sparsity)
                mean_eff_bits = sum(layer_eff_bits) / len(layer_eff_bits)

                logger.info(f"{layer_idx:<8} {q_sp:>4.1f}%    {k_sp:>4.1f}%    {v_sp:>4.1f}%    {o_sp:>4.1f}%    "
                           f"{down_sp:>4.1f}%    {gate_sp:>4.1f}%    {up_sp:>4.1f}%    "
                           f"{mean_sparsity*100:>5.2f}%       {mean_eff_bits:.3f}")

            # Summary statistics by module type
            module_type_stats = {}
            for layer_idx, modules in layer_stats.items():
                for module_type, stats in modules.items():
                    if module_type not in module_type_stats:
                        module_type_stats[module_type] = {'sparsity': [], 'eff_bits': []}
                    module_type_stats[module_type]['sparsity'].append(stats['sparsity'])
                    module_type_stats[module_type]['eff_bits'].append(stats['effective_bits'])

            logger.info("=" * 50)
            logger.info("Summary by Module Type:")
            logger.info(f"{'Module':<15} {'Avg Sparsity':<15} {'Avg Eff Bits':<12} {'Count':<8}")
            logger.info("-" * 50)
            for module_type in sorted(module_type_stats.keys()):
                avg_sp = sum(module_type_stats[module_type]['sparsity']) / len(module_type_stats[module_type]['sparsity'])
                avg_eb = sum(module_type_stats[module_type]['eff_bits']) / len(module_type_stats[module_type]['eff_bits'])
                count = len(module_type_stats[module_type]['sparsity'])
                logger.info(f"{module_type:<15} {avg_sp*100:>6.2f}%         {avg_eb:>6.3f}        {count:<8}")
            logger.info("=" * 80)
    else:
        logger.info("No QuantMoE modules found in the model.")
    if all_token_bit_distributions:
        logger.info("=" * 80)
        logger.info("Token Bit Distribution per Layer and Module:")
        logger.info(f"{'Layer':<8} {'Module':<12} {'Bit':<6} {'Percentage':<10}")
        logger.info("-" * 50)

        for layer_idx in sorted(layer_stats.keys()):
            for module_type, stats in layer_stats[layer_idx].items():
                token_dist = stats.get('token_bit_distribution', {})
                total_tokens = sum(token_dist.values())
                if total_tokens == 0:
                    continue

                for bit in sorted(token_dist.keys()):
                    count = token_dist[bit]
                    percentage = (count / total_tokens) * 100
                    logger.info(f"{layer_idx:<8} {module_type:<12} {bit:<6} {percentage:>6.2f}%")
        logger.info("=" * 80)

        # Optionally, you can also compute a global aggregated distribution
        aggregated_distribution = {}
        for layer_idx, modules in layer_stats.items():
            for module_type, stats in modules.items():
                token_dist = stats.get('token_bit_distribution', {})
                for bit, count in token_dist.items():
                    if bit not in aggregated_distribution:
                        aggregated_distribution[bit] = 0
                    aggregated_distribution[bit] += count

        logger.info("Aggregated Token Bit Distribution:")
        total_tokens = sum(aggregated_distribution.values())
        for bit in sorted(aggregated_distribution.keys()):
            count = aggregated_distribution[bit]
            percentage = (count / total_tokens) * 100
            logger.info(f"Bits: {bit}, Tokens: {count}, Percentage: {percentage:.2f}%")
        logger.info("=" * 80)


@torch.no_grad()
def calibrate_router_threshold(lm, args, dataloader, logger):
    """
    Calibrate a layer-wise global threshold for router scores to achieve the target bit-rate/sparsity.
    This respects the relative confidence of the trained router within each layer.
    """
    logger.info("Calibrating router threshold (Layer-wise)...")
    lm.model.eval()
    lm.model.config.use_cache = False

    # Move model to device before calibration
    if "opt" in args.net.lower():
        lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        layers = lm.model.model.decoder.layers
    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
        lm.model = lm.model.to(lm.device)
        layers = lm.model.model.layers
    elif "falcon" in args.net.lower():
        lm.model.transformer = lm.model.transformer.to(lm.device)
        layers = lm.model.transformer.h
    else:
        return

    layer_scores = {}  # {layer_idx: [tensor_scores, ...]}
    layer_modules = {}  # {layer_idx: [module1, module2, ...]}

    # 1. Collect scores from all QuantMoE modules across calibration samples (organized by layer)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Calibrating Threshold")):
            if i >= args.nsamples: break
            try:
                lm.model(batch[0].to(lm.device))
            except ValueError:
                pass

            for name, module in lm.model.named_modules():
                if isinstance(module, QuantMoE):
                    # Parse layer index from name (e.g., "model.layers.0.mlp.gate_proj" -> layer=0)
                    parts = name.split('.')
                    layer_idx = None
                    for j, part in enumerate(parts):
                        if part == 'layers' and j + 1 < len(parts):
                            layer_idx = int(parts[j + 1])
                            break

                    if layer_idx is not None:
                        if layer_idx not in layer_scores:
                            layer_scores[layer_idx] = []
                            layer_modules[layer_idx] = []

                        # Collect flattened scores (on CPU to save GPU memory)
                        layer_scores[layer_idx].append(module.Mask_Router.scores.detach().cpu().view(-1))
                        layer_modules[layer_idx].append(module)

    if not layer_scores:
        logger.warning("No scores collected for calibration.")
        return

    # 2. Calculate target ratio (fraction of experts that should be active globally)
    if args.target_activation_ratio_eval is not None:
        target_ratio = args.target_activation_ratio_eval
    elif args.target_bit_eval is not None:
        msb_bit = args.slice_bits_list[0]
        residual_bits_sum = sum(args.slice_bits_list[1:])
        target_ratio = (args.target_bit_eval - msb_bit) / residual_bits_sum
    else:
        return

    logger.info(f"Target Ratio: {target_ratio:.4f}")

    # 3. Find the per-layer threshold (quantile) and set it for all modules in that layer
    for layer_idx in sorted(layer_scores.keys()):
        # Concatenate all scores from all modules in this layer
        all_scores = torch.cat(layer_scores[layer_idx])

        # Use kthvalue for memory efficiency
        k = int((1 - target_ratio) * all_scores.numel())
        k = max(1, min(k, all_scores.numel()))
        threshold = torch.kthvalue(all_scores.float(), k).values.item()

        # Set the same threshold for all modules in this layer
        for module in layer_modules[layer_idx]:
            module.Mask_Router.global_threshold = threshold

        logger.info(f"Layer {layer_idx}: threshold = {threshold:.4f}")

    logger.info("Layer-wise thresholds calibrated.")


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))
    for dataset in ["wikitext2"]:
        # cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
        # if os.path.exists(cache_testloader):
        #     testloader = torch.load(cache_testloader)
        #     logger.info(f"load calibration from {cache_testloader}")
        # else:
        dataloader, testloader = get_loaders(
            dataset,
            seed=args.seed,
            model=args.model,
            seqlen=lm.seqlen,
        )
        # torch.save(testloader, cache_testloader)
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // lm.seqlen # ! chunked here. 
        use_cache = lm.model.config.use_cache
        lm.model.config.use_cache = False
        lm.model.eval()
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
            if "opt" in args.net.lower():
                outputs = lm.model.model.decoder(batch)
            elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                outputs = lm.model.model(batch)
            elif "falcon" in args.model:
                outputs = lm.model.transformer(batch)
            hidden_states = outputs[0]
            logits = lm.model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                :, 1:
            ].to(lm.model.lm_head.weight.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * lm.seqlen
            nlls.append(neg_log_likelihood)
            if i == args.limit:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
        logger.info(f'{dataset} : {ppl.item()}')
        lm.model.config.use_cache = use_cache
        results[dataset] = ppl.item()

        # Log to W&B
        if args.wandb:
            wandb.log({f"eval/{dataset}_ppl": ppl.item()})
                   
    return results


def main():
    from parser import args
    if args.onlymlp:
        from models.int_llama_layer_onlyMLP import QuantLlamaDecoderLayer
    else:
        from models.int_llama_layer import QuantLlamaDecoderLayer
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize W&B
    is_sweep_run = os.getenv('WANDB_SWEEP_ID') is not None
    if args.wandb and not is_sweep_run:
        wandb.init(
            project=args.wandb_project,
            entity="a2j-duke",
            config=vars(args),
            name=args.wandb_name
        )
    elif is_sweep_run:
        wandb.init()
        args.wandb = True  # Ensure logging happens during sweep
        # Override args with sweep parameters if running a sweep
        if wandb.config.get('model') is not None:
            args.model = wandb.config.model
        if wandb.config.get('output_dir') is not None:
            args.output_dir = wandb.config.output_dir
        if wandb.config.get('stage_weight_1') is not None:
            # stage_weight_1 is for stage_weight[1]
            args.stage_weight[1] = wandb.config.stage_weight_1
        if wandb.config.get('reg_weight') is not None:
            args.reg_weight = wandb.config.reg_weight
        if wandb.config.get('bitmoe_lr') is not None:
            args.bitmoe_lr = wandb.config.bitmoe_lr
        if wandb.config.get('target_activation_ratio') is not None:
            args.target_activation_ratio = wandb.config.target_activation_ratio

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    logger.info(f"attn block bits: {args.attnbits}")
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model.split("/", 1)[1]}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            # torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        bitmoequant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)

    # Run evaluation
    if args.slice_bits_list is not None and (args.target_bit_eval != 2.0):
        calibrate_router_threshold(lm, args, dataloader, logger)
        results = evaluate(lm, args, logger)
        collect_moe_stats(lm, args, dataloader, logger)
    else:
        results = evaluate(lm, args, logger)

    # Calculate combined score for multi-objective optimization
    if args.wandb:
        # Get PPL and effective bits from wandb logs
        ppl = results.get('wikitext2', None)
        # Try to get effective bits from wandb summary (set by collect_moe_stats)
        if wandb.run is not None and wandb.run.summary.get('eval/mean_effective_bits') is not None:
            mean_eff_bits = wandb.run.summary['eval/mean_effective_bits']
            # Normalize and combine (you can adjust the weights)
            # Combined score = normalized_ppl + normalized_eff_bits
            # Assuming PPL ~ 10-20 and eff_bits ~ 2-4
            if ppl is not None:
                normalized_ppl = ppl / 20.0  # normalize to ~0-1
                normalized_bits = mean_eff_bits / 4.0  # normalize to ~0-1
                combined_score = normalized_ppl + normalized_bits
                wandb.log({"eval/combined_score": combined_score})
                logger.info(f"Combined Score: {combined_score:.4f} (PPL: {ppl:.2f}, Eff Bits: {mean_eff_bits:.3f})")

    # Finish W&B run
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    print(sys.argv)
    main()
