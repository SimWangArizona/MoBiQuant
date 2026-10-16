"""Public model identities and portable default locations."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get('MOBIQUANT_MODEL_ROOT', ROOT / 'pretrained_models'))
CHECKPOINT_ROOT = Path(os.environ.get('MOBIQUANT_CHECKPOINT_ROOT', ROOT / 'checkpoints'))
CHECKPOINT_REPO = 'Simwang/MoBiQuant'
CHECKPOINT_REVISION = '6c1de7df7f0b9cb2292e95a087f69131b674f029'
PRESETS = {
    'llama2-7b': dict(folder='Llama-2-7b-hf', repo='meta-llama/Llama-2-7b-hf',
                     revision='01c7f73d771dfac7d292323805ebc428287df4f9'),
    'llama3-8b': dict(folder='Llama-3-8B', repo='meta-llama/Meta-Llama-3-8B',
                     revision='8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920'),
    'llama3-3b': dict(folder='Llama-3.2-3B', repo='meta-llama/Llama-3.2-3B',
                     revision='13afe5124825b4f3751f836b40dafda64c1ed062'),
    'llama3-1b': dict(folder='Llama-3.2-1B', repo='meta-llama/Llama-3.2-1B',
                     revision='4e20de362430cd3b72f300e6b0f18e50e7166e08'),
}
PARAMETER_FILES = ('omni_parameters.pth', 'router_parameters.pth')
