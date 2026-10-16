#!/usr/bin/env python3
"""Train at a 3-bit target or sweep checkpoint evaluation for four LLaMA models."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'llama2-7b': '/mnt/fast7T/models/Llama-2-7b-hf',
    'llama3-1b': '/mnt/fast7T/models/Llama-3.2-1B',
    'llama3-3b': '/mnt/fast7T/models/Llama-3.2-3B',
    'llama3-8b': '/mnt/fast7T/models/Llama-3-8B',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'sweep'])
    parser.add_argument('preset', choices=MODELS)
    parser.add_argument('--model', help='Override pretrained model path or Hugging Face ID')
    parser.add_argument('--checkpoint', type=Path, help='Default: log/<preset>-target3')
    parser.add_argument('--output', type=Path, help='Sweep output directory')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=-1, help='Original buildq2 limit behavior; use -1 for full/paper evaluation')
    parser.add_argument('--bits', type=float, nargs='+', default=[2, 2.5, 3, 3.5, 4, 5, 6, 7, 8])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.nsamples < 2 or args.seqlen < 2 or args.epochs < 1 or args.limit == 0 or args.limit < -1:
        parser.error('Require nsamples>=2, seqlen>=2, epochs>=1 and limit=-1 or positive')
    if any(not math.isfinite(bit) or not 2 <= bit <= 8 for bit in args.bits) or len(set(args.bits)) != len(args.bits):
        parser.error('Sweep bits must be distinct finite values in [2, 8]')
    checkpoint = (args.checkpoint or ROOT / 'log' / f'{args.preset}-target3').resolve()
    output = (args.output or ROOT / 'results' / args.preset / 'sweep').resolve()
    common = [sys.executable, str(ROOT / 'main_bitmoe.py'), '--model', args.model or MODELS[args.preset],
              '--wbits', '2', '--abits', '16', '--group_size', '128', '--lwc',
              '--slice_bits_list', '2', '2', '2', '2', '--nsamples', str(args.nsamples),
              '--seqlen', str(args.seqlen), '--limit', str(args.limit), '--eval_ppl']
    plans = []
    if args.mode == 'train':
        command = common + ['--epochs', str(args.epochs), '--batch_size', '1', '--target_bit', '3',
                            '--stage_weight', '1.0', '1.33', '--bitmoe_lr', '8e-5', '--reg_weight', '8e-5',
                            '--lwc_lr', '1e-3', '--output_dir', str(checkpoint),
                            '--metrics_file', str(checkpoint / 'train-metrics.json')]
        plans.append({'command': command, 'log': str(checkpoint / 'train.log')})
    else:
        for bit in args.bits:
            point = output / f'bits-{bit:g}'
            plans.append({'target_bit_eval': bit, 'metrics': str(point / 'metrics.json'),
                          'log': str(point / 'eval.log'),
                          'command': common + ['--epochs', '0', '--resume', str(checkpoint),
                                               '--target_bit_eval', str(bit), '--output_dir', str(point),
                                               '--metrics_file', str(point / 'metrics.json')]})
    if args.dry_run:
        print(json.dumps({'checkpoint': str(checkpoint), 'plans': plans}, indent=2))
        return
    if args.mode == 'train':
        if checkpoint.exists() and any(checkpoint.iterdir()):
            parser.error(f'Refusing to overwrite nonempty training directory: {checkpoint}')
    else:
        for name in ['omni_parameters.pth', 'router_parameters.pth', 'router_stats.pth']:
            if not (checkpoint / name).is_file():
                parser.error(f'Missing checkpoint file: {checkpoint / name}')
        if output.exists() and any(output.iterdir()):
            parser.error(f'Refusing to overwrite nonempty sweep directory: {output}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    csv_file = None
    try:
        if args.mode == 'sweep':
            output.mkdir(parents=True, exist_ok=True)
            csv_file = (output / 'bits-performance.csv').open('w', newline='')
            writer = csv.DictWriter(csv_file, fieldnames=['target_bit_eval', 'actual_bits', 'wikitext2_ppl', 'moe_modules', 'status', 'metrics_file', 'log_file'])
            writer.writeheader(); csv_file.flush()
        for plan in plans:
            log = Path(plan['log']); log.parent.mkdir(parents=True, exist_ok=True)
            print('Running:', ' '.join(plan['command']), flush=True)
            with log.open('w') as stream:
                result = subprocess.run(plan['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if args.mode == 'train':
                if result.returncode:
                    raise RuntimeError(f'Training failed; see {log}')
                for name in ['omni_parameters.pth', 'router_parameters.pth', 'router_stats.pth']:
                    if not (checkpoint / name).is_file():
                        raise RuntimeError(f'Training did not create {name}; see {log}')
                print(f'Trained checkpoint: {checkpoint}')
                continue
            row = dict(target_bit_eval=plan['target_bit_eval'], actual_bits='', wikitext2_ppl='',
                       moe_modules='', status='failed', metrics_file=plan['metrics'], log_file=plan['log'])
            try:
                if result.returncode:
                    raise RuntimeError(f'Evaluation exited with code {result.returncode}')
                metrics = json.loads(Path(plan['metrics']).read_text())
                actual = float(metrics['actual_bits']); ppl = float(metrics['wikitext2'])
                if not (math.isfinite(actual) and 2 <= actual <= 8 and math.isfinite(ppl) and ppl > 0):
                    raise ValueError('Invalid bits/PPL metrics')
                row.update(actual_bits=actual, wikitext2_ppl=ppl, moe_modules=metrics['moe_modules'], status='ok')
            finally:
                writer.writerow(row); csv_file.flush()
            print(row, flush=True)
        if csv_file:
            print(f'Results: {output / "bits-performance.csv"}')
    finally:
        if csv_file:
            csv_file.close()


if __name__ == '__main__':
    main()
