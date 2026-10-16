"""Reduced real-weight WikiText validation; not production-quality training."""
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
preset=sys.argv[1]; gpu=sys.argv[2]
checkpoint=Path('/tmp/mobiquant-audit-runs') / preset / 'checkpoint'
output=checkpoint.parent/'sweep'
common=[sys.executable,str(root/'scripts/bitmoe/run.py')]
env=dict(os.environ,HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
options=['--gpu',gpu,'--nsamples','2','--seqlen','32','--limit','2','--checkpoint',str(checkpoint)]
subprocess.run(common+['train',preset,'--epochs','1']+options,cwd=root,env=env,check=True)
subprocess.run(common+['sweep',preset,'--bits','2','3','8','--output',str(output)]+options,cwd=root,env=env,check=True)
print('PASS',preset, 'train target=3 and sweep endpoints/interior:',output/'bits-performance.csv')
