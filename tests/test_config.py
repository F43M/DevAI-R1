import os
import tempfile
import config_utils
from config_utils import load_config

def test_load_config_merges_defaults():
    defaults = {'A': 1, 'B': 2}
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, 'config.yaml')
        with open(cfg_path, 'w') as f:
            f.write('A: 3')
        cfg = load_config(cfg_path, defaults)
        if config_utils.yaml is None:
            assert cfg == defaults
        else:
            assert cfg['A'] == 3
            assert cfg['B'] == 2
