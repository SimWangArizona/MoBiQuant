"""Validate all four presets and prevent stale/failed sweep rows."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
runner=ROOT/'scripts/bitmoe/run.py'
for preset in ['llama2-7b','llama3-1b','llama3-3b','llama3-8b']:
    for mode in ['train','sweep']:
        result=json.loads(subprocess.check_output([sys.executable,str(runner),mode,preset,'--dry-run'],text=True))
        for plan in result['plans']:
            c=plan['command']
            assert '--onlymlp' not in c
            assert c[c.index('--slice_bits_list')+1:c.index('--slice_bits_list')+5]==['2']*4
            assert '--metrics_file' in c
            if mode=='train':
                assert c[c.index('--target_bit')+1]=='3'
                assert '--target_activation_ratio' not in c
            else:
                assert c[c.index('--epochs')+1]=='0' and '--resume' in c
spec=importlib.util.spec_from_file_location('runner',runner)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for code in [1,0]:
    with tempfile.TemporaryDirectory() as temp:
        checkpoint=Path(temp)/'checkpoint';checkpoint.mkdir()
        for name in ['omni_parameters.pth','router_parameters.pth','router_stats.pth']:
            (checkpoint/name).touch()
        output=Path(temp)/'sweep'
        argv=[str(runner),'sweep','llama3-1b','--bits','3','--checkpoint',str(checkpoint),'--output',str(output)]
        with patch.object(sys,'argv',argv), patch.object(module.subprocess,'run',return_value=SimpleNamespace(returncode=code)):
            try:module.main()
            except (RuntimeError,FileNotFoundError):pass
            else:raise AssertionError('Accepted failed evaluation or missing metrics')
        rows=list(csv.DictReader((output/'bits-performance.csv').open()))
        assert len(rows)==1 and rows[0]['status']=='failed' and rows[0]['actual_bits']=='' and rows[0]['wikitext2_ppl']==''
print('PASS four target3/MoE presets; failed subprocess and missing metrics produce failed rows without fabricated bits/PPL')
