"""Check completed GPU runs, verify every layer's seven routers, archive small results."""
import csv
import json
from pathlib import Path
import shutil
import torch

ROOT=Path(__file__).resolve().parents[1]
RUNS=Path('/tmp/mobiquant-audit-runs')
MODELS={'llama2-7b':('/mnt/fast7T/models/Llama-2-7b-hf',32),
        'llama3-1b':('/mnt/fast7T/models/Llama-3.2-1B',16),
        'llama3-3b':('/mnt/fast7T/models/Llama-3.2-3B',28),
        'llama3-8b':('/mnt/fast7T/models/Llama-3-8B',32)}
summary=[]
for preset,(model,layers) in MODELS.items():
    run=RUNS/preset
    rows=list(csv.DictReader((run/'sweep/bits-performance.csv').open()))
    assert len(rows)==3 and {float(row['target_bit_eval']) for row in rows}=={2,3,8},preset
    for row in rows:
        assert row['status']=='ok' and int(row['moe_modules'])==layers*7,row
    assert float(rows[0]['actual_bits'])==2
    train=json.loads((run/'checkpoint/train-metrics.json').read_text())
    assert train['target_bit_train']==3 and train['moe_modules']==layers*7
    routers=torch.load(run/'checkpoint/router_parameters.pth',map_location='cpu',mmap=True)
    assert set(routers)==set(range(layers)),preset
    prefixes=['self_attn.'+n+'_proj' for n in ['q','k','v','o']]+['mlp.'+n+'_proj' for n in ['gate','up','down']]
    for layer in routers.values():
        for prefix in prefixes:
            for index in [0,1]:
                assert f'{prefix}.Mask_Router.router.{index}.weight' in layer,(preset,prefix)
    destination=ROOT/'validation/real-model-results'/preset
    destination.mkdir(parents=True,exist_ok=True)
    for row in rows:
        bit = float(row['target_bit_eval'])
        row['metrics_file'] = f'bits-{bit:g}/metrics.json'
        row['log_file'] = f'bits-{bit:g}/eval.log'
    with (destination/'bits-performance.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    shutil.copy2(run/'checkpoint/train-metrics.json',destination/'train-metrics.json')
    shutil.copy2(run/'checkpoint/train.log',destination/'train.log')
    for bit in [2,3,8]:
        point = destination/f'bits-{bit}'
        point.mkdir(exist_ok=True)
        for filename in ['metrics.json', 'eval.log']:
            shutil.copy2(run/f'sweep/bits-{bit}'/filename,point/filename)
    summary.append(dict(preset=preset,model=model,layers=layers,attention_moe_modules=layers*4,
                        mlp_moe_modules=layers*3,training_target_bits=3,sweep_points=[2,3,8],status='passed',
                        epochs=1,calibration_samples=2,seqlen=32,ppl_sequences=2))
(ROOT/'validation/real-model-results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for log in (ROOT/'validation/real-model-results').rglob('*.log'):
    log.write_text('\n'.join(line.rstrip() for line in log.read_text().splitlines()) + '\n')
print('PASS four real models: complete layer router coverage, target3 training, three successful sweep rows per model')
