import os
import json
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
import torch.nn as nn
from quantize.bitmoequant_llama3 import bitmoequant
from tqdm import tqdm
import utils
from pathlib import Path
from quantize.int_moe import QuantMoE
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def collect_moe_stats(lm, args, dataloader, logger):
    """Collect sparsity and effective bits from all QuantMoE modules after running forward passes.

    This function:
    1. Runs forward passes on calibration data to update router statistics
    2. Collects sparsity and effective bits from all QuantMoE modules
    3. Logs the statistics
    """
    logger.info('Collecting QuantMoE statistics from calibration data...')
    lm.model.eval()
    dev = lm.device
    layers = lm.model.model.layers
    for name, module in lm.model.named_modules():
        if isinstance(module, QuantMoE):
            module.Mask_Router.to(dev)
    for i, batch in enumerate(dataloader):
        if i >= args.nsamples:
            break
        _ = lm.model(batch[0].to(dev))
    all_sparsity = []
    all_eff_bits = []
    all_token_bit_distributions = []
    layer_stats = {}
    for name, module in lm.model.named_modules():
        if isinstance(module, QuantMoE):
            sparsity = module.get_sparsity()
            eff_bits = module.get_effective_bits()
            token_bit_distribution = module.get_token_bit_distribution()
            all_sparsity.append(sparsity)
            all_eff_bits.append(eff_bits)
            all_token_bit_distributions.append(token_bit_distribution)
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
        logger.info('=' * 80)
        logger.info(f'QuantMoE Statistics (collected over {args.nsamples} samples):')
        logger.info(f'Total QuantMoE modules: {len(all_sparsity)}')
        logger.info(f'Average Sparsity: {avg_sparsity:.4f} ({avg_sparsity * 100:.2f}%)')
        logger.info(f'Average Effective Bits: {avg_eff_bits:.3f}')
        logger.info('=' * 80)
        if layer_stats:
            logger.info('Layer-wise Statistics:')
            logger.info(f"{'Layer':<8} {'Q':<8} {'K':<8} {'V':<8} {'O':<8} {'Down':<8} {'Gate':<8} {'Up':<8} {'Mean Spar':<12} {'Mean Bits':<10}")
            logger.info('-' * 100)
            for layer_idx in sorted(layer_stats.keys()):
                q_sp = layer_stats[layer_idx].get('q_proj', {}).get('sparsity', 0) * 100
                k_sp = layer_stats[layer_idx].get('k_proj', {}).get('sparsity', 0) * 100
                v_sp = layer_stats[layer_idx].get('v_proj', {}).get('sparsity', 0) * 100
                o_sp = layer_stats[layer_idx].get('o_proj', {}).get('sparsity', 0) * 100
                down_sp = layer_stats[layer_idx].get('down_proj', {}).get('sparsity', 0) * 100
                gate_sp = layer_stats[layer_idx].get('gate_proj', {}).get('sparsity', 0) * 100
                up_sp = layer_stats[layer_idx].get('up_proj', {}).get('sparsity', 0) * 100
                layer_sparsity = [stats['sparsity'] for stats in layer_stats[layer_idx].values()]
                layer_eff_bits = [stats['effective_bits'] for stats in layer_stats[layer_idx].values()]
                mean_sparsity = sum(layer_sparsity) / len(layer_sparsity)
                mean_eff_bits = sum(layer_eff_bits) / len(layer_eff_bits)
                logger.info(f'{layer_idx:<8} {q_sp:>4.1f}%    {k_sp:>4.1f}%    {v_sp:>4.1f}%    {o_sp:>4.1f}%    {down_sp:>4.1f}%    {gate_sp:>4.1f}%    {up_sp:>4.1f}%    {mean_sparsity * 100:>5.2f}%       {mean_eff_bits:.3f}')
            module_type_stats = {}
            for layer_idx, modules in layer_stats.items():
                for module_type, stats in modules.items():
                    if module_type not in module_type_stats:
                        module_type_stats[module_type] = {'sparsity': [], 'eff_bits': []}
                    module_type_stats[module_type]['sparsity'].append(stats['sparsity'])
                    module_type_stats[module_type]['eff_bits'].append(stats['effective_bits'])
            logger.info('=' * 50)
            logger.info('Summary by Module Type:')
            logger.info(f"{'Module':<15} {'Avg Sparsity':<15} {'Avg Eff Bits':<12} {'Count':<8}")
            logger.info('-' * 50)
            for module_type in sorted(module_type_stats.keys()):
                avg_sp = sum(module_type_stats[module_type]['sparsity']) / len(module_type_stats[module_type]['sparsity'])
                avg_eb = sum(module_type_stats[module_type]['eff_bits']) / len(module_type_stats[module_type]['eff_bits'])
                count = len(module_type_stats[module_type]['sparsity'])
                logger.info(f'{module_type:<15} {avg_sp * 100:>6.2f}%         {avg_eb:>6.3f}        {count:<8}')
            logger.info('=' * 80)
    else:
        logger.info('No QuantMoE modules found in the model.')
    if all_token_bit_distributions:
        logger.info('=' * 80)
        logger.info('Token Bit Distribution per Layer and Module:')
        logger.info(f"{'Layer':<8} {'Module':<12} {'Bit':<6} {'Percentage':<10}")
        logger.info('-' * 50)
        for layer_idx in sorted(layer_stats.keys()):
            for module_type, stats in layer_stats[layer_idx].items():
                token_dist = stats.get('token_bit_distribution', {})
                total_tokens = sum(token_dist.values())
                if total_tokens == 0:
                    continue
                for bit in sorted(token_dist.keys()):
                    count = token_dist[bit]
                    percentage = count / total_tokens * 100
                    logger.info(f'{layer_idx:<8} {module_type:<12} {bit:<6} {percentage:>6.2f}%')
        logger.info('=' * 80)
        aggregated_distribution = {}
        for layer_idx, modules in layer_stats.items():
            for module_type, stats in modules.items():
                token_dist = stats.get('token_bit_distribution', {})
                for bit, count in token_dist.items():
                    if bit not in aggregated_distribution:
                        aggregated_distribution[bit] = 0
                    aggregated_distribution[bit] += count
        logger.info('Aggregated Token Bit Distribution:')
        total_tokens = sum(aggregated_distribution.values())
        for bit in sorted(aggregated_distribution.keys()):
            count = aggregated_distribution[bit]
            percentage = count / total_tokens * 100
            logger.info(f'Bits: {bit}, Tokens: {count}, Percentage: {percentage:.2f}%')
        logger.info('=' * 80)

@torch.no_grad()
def calibrate_router_threshold(lm, args, dataloader, logger):
    """
    Calibrate a layer-wise global threshold for router scores to achieve the target bit-rate/sparsity.
    This respects the relative confidence of the trained router within each layer.
    """
    logger.info('Calibrating router threshold (Layer-wise)...')
    lm.model.eval()
    lm.model.config.use_cache = False
    lm.model = lm.model.to(lm.device)
    layers = lm.model.model.layers
    layer_scores = {}
    layer_modules = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Calibrating Threshold')):
            if i >= args.nsamples:
                break
            try:
                lm.model(batch[0].to(lm.device))
            except ValueError:
                pass
            for name, module in lm.model.named_modules():
                if isinstance(module, QuantMoE):
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
                        layer_scores[layer_idx].append(module.Mask_Router.scores.detach().cpu().view(-1))
                        layer_modules[layer_idx].append(module)
    if not layer_scores:
        logger.warning('No scores collected for calibration.')
        return
    if args.target_activation_ratio_eval is not None:
        target_ratio = args.target_activation_ratio_eval
    elif args.target_bit_eval is not None:
        msb_bit = args.slice_bits_list[0]
        residual_bits_sum = sum(args.slice_bits_list[1:])
        target_ratio = (args.target_bit_eval - msb_bit) / residual_bits_sum
    else:
        return
    logger.info(f'Target Ratio: {target_ratio:.4f}')
    for layer_idx in sorted(layer_scores.keys()):
        all_scores = torch.cat(layer_scores[layer_idx])
        k = int((1 - target_ratio) * all_scores.numel())
        k = max(1, min(k, all_scores.numel()))
        threshold = torch.kthvalue(all_scores.float(), k).values.item()
        for module in layer_modules[layer_idx]:
            module.Mask_Router.global_threshold = threshold
        logger.info(f'Layer {layer_idx}: threshold = {threshold:.4f}')
    logger.info('Layer-wise thresholds calibrated.')

@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}
    lm.model = lm.model.to(lm.device)
    for dataset in ['wikitext2']:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=lm.seqlen)
        if 'c4' in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids
        nsamples = testenc.numel() // lm.seqlen
        use_cache = lm.model.config.use_cache
        lm.model.config.use_cache = False
        lm.model.eval()
        nlls = []
        for i in tqdm(range(nsamples)):
            batch = testenc[:, i * lm.seqlen:(i + 1) * lm.seqlen].to(lm.device)
            outputs = lm.model.model(batch)
            hidden_states = outputs[0]
            logits = lm.model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, i * lm.seqlen:(i + 1) * lm.seqlen][:, 1:].to(lm.model.lm_head.weight.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * lm.seqlen
            nlls.append(neg_log_likelihood)
            if i == args.limit:
                break
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
        logger.info(f'{dataset} : {ppl.item()}')
        lm.model.config.use_cache = use_cache
        results[dataset] = ppl.item()
    return results

def main():
    from parser import args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.epochs > 0:
        assert args.lwc or args.let
    if args.wbits < 16 and args.wbits >= 8 or (args.abits < 16 and args.abits >= 8):
        args.deactive_amp = True
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    logger.info('Weight quantization config: wbits=%s, slice_bits_list=%s; activation bits=%s', args.wbits, args.slice_bits_list, args.abits)
    if args.net is None:
        args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = args.seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    args.weight_quant_params = {'n_bits': args.wbits, 'per_channel_axes': [0], 'symmetric': args.symmetric, 'dynamic_method': args.w_dynamic_method, 'group_size': args.group_size, 'lwc': args.lwc, 'disable_zero_point': args.disable_zero_point}
    args.act_quant_params = {'n_bits': args.abits, 'per_channel_axes': [], 'symmetric': False, 'dynamic_method': args.a_dynamic_method}
    args.q_quant_params = {'n_bits': args.abits, 'per_channel_axes': [], 'symmetric': False, 'dynamic_method': args.a_dynamic_method}
    args.k_quant_params = {'n_bits': args.abits, 'per_channel_axes': [], 'symmetric': False, 'dynamic_method': args.a_dynamic_method}
    args.v_quant_params = {'n_bits': args.abits, 'per_channel_axes': [], 'symmetric': False, 'dynamic_method': args.a_dynamic_method}
    args.p_quant_params = {'n_bits': 16, 'metric': 'fix0to1'}
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'
    if args.wbits < 16 or args.abits < 16:
        logger.info('=== start quantization ===')
        tick = time.time()
        cache_dataloader = f"{args.cache_dir}/dataloader_{args.model.split('/', 1)[1]}_{args.calib_dataset}_{args.nsamples}.cache"
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f'load calibration from {cache_dataloader}')
        else:
            dataloader, _ = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=lm.seqlen)
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        bitmoequant(lm, args, dataloader, act_scales, act_shifts, logger)
        logger.info(time.time() - tick)
    if args.slice_bits_list is not None and args.target_bit_eval != 2.0:
        calibrate_router_threshold(lm, args, dataloader, logger)
        results = evaluate(lm, args, logger)
        collect_moe_stats(lm, args, dataloader, logger)
        moe_modules = [m for m in lm.model.modules() if isinstance(m, QuantMoE)]
        if moe_modules:
            results.update(actual_bits=sum((m.get_effective_bits() for m in moe_modules)) / len(moe_modules), moe_modules=len(moe_modules), bits_measurement='original buildq2 statistics: last calibration sample, mean over projections')
    else:
        results = evaluate(lm, args, logger)
        if args.slice_bits_list is not None:
            results.update(actual_bits=float(args.slice_bits_list[0]), moe_modules=sum((isinstance(m, QuantMoE) for m in lm.model.modules())), bits_measurement='MSB-only; residual experts disabled')
    if args.metrics_file:
        results.update(model=args.model, target_bit_eval=args.target_bit_eval, target_bit_train=args.target_bit, seqlen=lm.seqlen)
        metrics_path = Path(args.metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metrics_path.with_suffix(metrics_path.suffix + '.tmp')
        temporary_path.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
        temporary_path.replace(metrics_path)
if __name__ == '__main__':
    print(sys.argv)
    main()
