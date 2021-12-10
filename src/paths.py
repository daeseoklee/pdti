from json import load as load_json
from collections import namedtuple
from pathlib import Path
from typing import Dict


BASE = Path(__file__).parent.parent

with open(BASE / 'paths.json', 'r') as reader:
    paths_dict = load_json(reader)
    paths_dict : Dict[str, str]
    paths = {}
    for group in paths_dict.values():
        for name, path in group.items():
            paths[name] = path

    
for key, val in paths.items():
    val : str
    if val.startswith('.') and val[1] in ['/', '\\']:
        new_val = val.replace('.', str(BASE), 1)
        paths[key] = new_val
paths = namedtuple('Paths', paths.keys())(**paths)

if __name__ == '__main__':
    print(paths.uniref100)
    print(paths.pr_vocab)
    print(paths.sm_vocab)
