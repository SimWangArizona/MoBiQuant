#!/usr/bin/env python3
"""Evaluate published or locally trained parameters from 2.0 to 8.0 bits in 0.1-bit steps."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows._config import (PRESETS, MODEL_ROOT, CHECKPOINT_ROOT, CHECKPOINT_REPO,
                     CHECKPOINT_REVISION, PARAMETER_FILES)


BITS = [value / 10 for value in range(20, 81)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preset', choices=PRESETS)
    parser.add_argument('--model', help='Override original base-model path or Hugging Face ID')
    parser.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT,
                        help='Local download directory for Simwang/MoBiQuant')
    parser.add_argument('--checkpoint', type=Path,
                        help='Use a locally trained parameter directory instead of downloading HF parameters')
    parser.add_argument('--revision', default=CHECKPOINT_REVISION, help='Simwang/MoBiQuant revision')
    parser.add_argument('--output', type=Path, help='Default: results/<preset>/finegrained-sweep-eval')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--limit', type=int, default=-1, help='Keep -1 for full evaluation')
    parser.add_argument('--dry-run', action='store_true', help='Print download and evaluation plans; no network or GPU work')
    args = parser.parse_args()
    if args.nsamples < 2 or args.seqlen < 2 or args.limit == 0 or args.limit < -1:
        parser.error('Require nsamples>=2, seqlen>=2 and limit=-1 or positive')
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else checkpoint_root / args.preset
    output = (args.output or ROOT / 'results' / args.preset / 'finegrained-sweep-eval').resolve()
    model = args.model or str((args.model_root / PRESETS[args.preset]['folder']).resolve())
    download = dict(repo_id=CHECKPOINT_REPO, revision=args.revision,
                    local_dir=str(checkpoint_root),
                    allow_patterns=[f'{args.preset}/{name}' for name in PARAMETER_FILES]) if not args.checkpoint else None
    common = [sys.executable, str(ROOT / 'main_bitmoe.py'), '--model', model,
              '--epochs', '0', '--resume', str(checkpoint),
              '--wbits', '2', '--abits', '16', '--group_size', '128', '--lwc',
              '--slice_bits_list', '2', '2', '2', '2', '--seed', '2',
              '--calib_dataset', 'wikitext2', '--batch_size', '1',
              '--cache_dir', str(args.cache_dir.resolve()),
              '--nsamples', str(args.nsamples), '--seqlen', str(args.seqlen),
              '--limit', str(args.limit), '--eval_ppl']
    plans = []
    for bit in BITS:
        point = output / f'bits-{bit:g}'
        metrics = point / 'metrics.json'
        plans.append(dict(target_bit_eval=bit, metrics=str(metrics), log=str(point / 'eval.log'),
                          command=common + ['--target_bit_eval', str(bit),
                                            '--output_dir', str(point), '--metrics_file', str(metrics)]))
    if args.dry_run:
        print(json.dumps(dict(download=download, checkpoint=str(checkpoint), plans=plans), indent=2))
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Refusing to overwrite nonempty sweep output: {output}')
    if download is not None:
        from huggingface_hub import snapshot_download
        print(f'Downloading {CHECKPOINT_REPO} ({args.preset}) ...', flush=True)
        snapshot_download(**download)
    for name in PARAMETER_FILES:
        if not (checkpoint / name).is_file():
            parser.error(f'Missing parameter file: {checkpoint / name}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'sweep-plan.json').write_text(json.dumps(dict(download=download, plans=plans), indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    csv_path = output / 'bits-performance.csv'
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            'target_bit_eval', 'actual_bits', 'wikitext2_ppl', 'moe_modules',
            'status', 'metrics_file', 'log_file'])
        writer.writeheader()
        csv_file.flush()
        for index, plan in enumerate(plans, 1):
            log = Path(plan['log'])
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f'Point {index}/{len(plans)}: {plan["target_bit_eval"]:.1f} bits', flush=True)
            print('Running:', shlex.join(plan['command']), flush=True)
            row = dict(target_bit_eval=plan['target_bit_eval'], actual_bits='', wikitext2_ppl='',
                       moe_modules='', status='failed', metrics_file=plan['metrics'], log_file=plan['log'])
            try:
                with log.open('w') as stream:
                    result = subprocess.run(plan['command'], cwd=ROOT, env=env,
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Evaluation exited with code {result.returncode}; see {log}')
                metrics = json.loads(Path(plan['metrics']).read_text())
                actual = float(metrics['actual_bits'])
                ppl = float(metrics['wikitext2'])
                if not (math.isfinite(actual) and 2 <= actual <= 8 and math.isfinite(ppl) and ppl > 0):
                    raise ValueError(f'Invalid bits/PPL metrics; see {log}')
                row.update(actual_bits=actual, wikitext2_ppl=ppl,
                           moe_modules=metrics['moe_modules'], status='ok')
            finally:
                writer.writerow(row)
                csv_file.flush()
            print(row, flush=True)
    print(f'Results: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
