from os import getcwd
from pathlib import Path 
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt
from alphafold_proteins import Atom 

def load_pickle(file):
    with open(str(file), 'rb') as f:
        return pickle.load(f)
CWD = Path(getcwd())


@dataclass
class Vertex:
    center: np.ndarray
    radius: float
    atoms: List

@dataclass
class Pocket:
    atoms: List[Atom]
    verts: List[Vertex]
    pocket_score: float
    drug_score: float
    volume: float

class NoPocket(Exception):
    def __init__(self):
        pass

def parse_fpocket_output(dir:Path, skip_atoms=False, min_cluster_size=None):
    def parse_atm_file(atm_file):
        atoms = [] 
        with open(str(atm_file), 'r') as reader:
            for line in reader.readlines():
                if line.startswith('ATOM '):
                    atom = Atom.parse(line)
                    atoms.append(atom)
        assert len(atoms) > 0
        return atoms

    def parse_vert_file(vert_file):
        verts = [] 
        info = {} 
        info_keys = {
            'Pocket Score': 'pocket_score',
            'Drug Score': 'drug_score',
            'Number of V. Vertices': 'size',
            'Real volume (approximation)': 'volume',
            'Hydrophobicity Score': 'hydrophobicity'
        }
        with open(str(vert_file), 'r') as reader:
            for line in reader.readlines():
                if line.startswith('HEADER ') and '-' in line:
                    dash_idx = line.find('-')
                    _key, _val = line[dash_idx+1:].split(':')
                    _key = _key.strip()
                    if not _key in info_keys:
                        continue
                    key = info_keys[_key]
                    if '.' in _val:
                        val = float(_val)
                    else:
                        val = int(_val)
                    
                    if key == 'size' and min_cluster_size is not None and val < min_cluster_size:
                        return None, None
                    
                    info[key] = val
                elif line.startswith('ATOM '):
                    assert line[17:20] == 'STP'
                    center = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    radius = float(line[66:])
                    vert = Vertex(center, radius, [])
                    verts.append(vert)
        assert len(info) == len(info_keys)
        assert len(verts) > 0 
        return verts, info


    pocketdir = dir / 'pockets'
    n = len(list(pocketdir.glob('pocket*_atm.pdb')))
    if n == 0:
        raise NoPocket
    pockets = [] 
    for i in range(1, n + 1):
        atm_file = pocketdir / f'pocket{i}_atm.pdb'
        vert_file = pocketdir / f'pocket{i}_vert.pqr'
        verts, info = parse_vert_file(vert_file)
        if info is None: #exceptional cases
            continue
        if skip_atoms:
            atoms = None
        else:
            atoms = parse_atm_file(atm_file)
        for vert in verts:
            if skip_atoms:
                vert.atoms = None
                continue
            for atom in atoms:
                dist = np.linalg.norm(atom.coord - vert.center)
                if abs(dist - vert.radius) < 0.02:
                    vert.atoms.append(atom)
        pocket = Pocket(atoms, verts, info['pocket_score'], info['drug_score'], info['volume'])
        pockets.append(pocket)
    return pockets
