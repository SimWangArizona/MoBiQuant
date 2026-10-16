"""Load a BitMoE parameter pair, with optional historical training statistics."""
from pathlib import Path
import torch


def load_moe_checkpoint(directory, epochs, logger):
    directory = Path(directory)
    omni = torch.load(directory / 'omni_parameters.pth', map_location='cpu', weights_only=True)
    router = torch.load(directory / 'router_parameters.pth', map_location='cpu', weights_only=True)
    stats_file = directory / 'router_stats.pth'
    if stats_file.is_file():
        stats = torch.load(stats_file, map_location='cpu', weights_only=True)
        global_steps = stats['config']['global_steps']
    elif epochs == 0:
        # Evaluation uses hard gates on raw scores. With step=None the sigmoid
        # temperature is already maximal regardless of this schedule length.
        stats = None
        global_steps = 1
        logger.info('Loaded parameter pair without router_stats.pth (evaluation only)')
    else:
        raise FileNotFoundError(f'Training resume requires {stats_file}')
    return omni, router, stats, global_steps
