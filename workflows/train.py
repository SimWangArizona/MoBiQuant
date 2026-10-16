#!/usr/bin/env python3
"""Train using the recipes recorded beside the four released checkpoints.

Example: bash scripts/llama3-1b/train.sh --gpu 0 --dry-run
Outputs go to log/retrained-<preset>, followed by a 61-point sweep of the new parameters.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows._config import MODEL_ROOT, PRESETS


RECIPES = json.loads(Path(__file__).with_name('training_recipes.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preset', choices=RECIPES)
    parser.add_argument('--model', help='Override the local base-model path with a path or HF ID')
    parser.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    parser.add_argument('--output', type=Path, help='Default: log/retrained-<preset>')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache')
    parser.add_argument('--sweep-output', type=Path, help='Default: <training output>/finegrained-sweep-eval')
    parser.add_argument('--skip-sweep', action='store_true', help='Only train and save parameters')
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    recipe = RECIPES[options.preset]
    output = (options.output or ROOT / 'log' / f'retrained-{options.preset}').resolve()
    model = options.model or str((options.model_root / PRESETS[options.preset]['folder']).resolve())
    sweep_output = (options.sweep_output or output / 'finegrained-sweep-eval').resolve()
    command = [sys.executable, str(ROOT / 'main_bitmoe.py'),
               '--model', model,
               '--output_dir', str(output), '--cache_dir', str(options.cache_dir.resolve()),
               '--seqlen', str(recipe['seqlen']), '--metrics_file', str(output / 'train-metrics.json')]
    for key, value in recipe['training_args'].items():
        if value is None or value is False:
            continue
        command.append('--' + key)
        if value is not True:
            command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    sweep_command = None if options.skip_sweep else [
        sys.executable, str(ROOT / 'workflows/finegrained_sweep_eval.py'), options.preset,
        '--model', model, '--checkpoint', str(output), '--output', str(sweep_output),
        '--gpu', options.gpu, '--nsamples', str(recipe['training_args']['nsamples']),
        '--cache-dir', str(options.cache_dir.resolve()),
        '--seqlen', str(recipe['seqlen']), '--limit', '-1',
    ]
    plan = dict(preset=options.preset, source_log=recipe['source_log'],
                output=str(output), gpu=options.gpu, command=command,
                sweep_command=sweep_command,
                adjustments=['Use local base-model path unless overridden',
                             'Use a new output directory; no resume',
                             recipe['seqlen_source']])
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f'Refusing to overwrite nonempty output: {output}')
    if sweep_command:
        if sweep_output == output or sweep_output in output.parents:
            parser.error('Sweep output must not equal or contain the training output directory')
        if sweep_output.exists() and (not sweep_output.is_dir() or any(sweep_output.iterdir())):
            parser.error(f'Refusing to overwrite nonempty sweep output: {sweep_output}')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'recipe.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('Recipe source:', recipe['source_log'], flush=True)
    print('Running:', shlex.join(command), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=options.gpu, PYTHONUNBUFFERED='1')
    # Entry-point logger writes its normal logs in output; retain stdout/stderr too.
    with (output / 'train.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        parser.exit(1, f'Training failed (exit {result.returncode}); see {output / "train.log"}\n')
    for name in ('omni_parameters.pth', 'router_parameters.pth', 'router_stats.pth'):
        if not (output / name).is_file():
            parser.exit(1, f'Training did not produce {name}; see {output / "train.log"}\n')
    print(f'Checkpoint saved: {output}', flush=True)
    if sweep_command:
        print('Starting 61-point sweep of the newly trained parameters...', flush=True)
        print('Running:', shlex.join(sweep_command), flush=True)
        with (output / 'sweep.log').open('w') as log:
            result = subprocess.run(sweep_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            parser.exit(1, f'Sweep failed (exit {result.returncode}); trained parameters are saved in {output}. '
                        f'See {output / "sweep.log"}\n')
        print(f'Sweep results: {sweep_output / "bits-performance.csv"}', flush=True)


if __name__ == '__main__':
    main()
