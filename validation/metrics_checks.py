"""Ensure paper-sensitive evaluation functions match original buildq2 exactly."""
import ast
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=subprocess.check_output(['git','show','1224843:main_bitmoe.py'],cwd=root,text=True)
current=(root/'main_bitmoe.py').read_text()
for name in ['evaluate','collect_moe_stats','calibrate_router_threshold']:
    def extract(text):
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
        return ''.join(text.splitlines(keepends=True)[start:node.end_lineno])
    assert extract(current)==extract(original),name
    print('PASS verbatim original buildq2 function:',name)
