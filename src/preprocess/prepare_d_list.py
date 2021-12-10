from typing import Tuple, List, Dict, Any
import pickle, json
import numpy as np 
import torch
from pathlib import Path
from paths import paths

from alphafold_proteins import ProteinInfo, Chain, Residue, Atom
from fpocket_outputs import parse_fpocket_output, NoPocket, Pocket
from frames import get_rotation
 
convert_residue={
'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',
'MSE':'M', 'ASX':'B', 'UNK' : 'X', 'SEC':'U','PYL':'O'
}

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def gen_name_seq_pairs(dataset):
    if dataset in ['kiba', 'davis']:
        proteins_filename = str(Path(paths.deepdta) / 'data' / dataset / 'proteins.txt')
        d = load_json(proteins_filename)
        for accession in d.keys():
            yield accession, d[accession]
    else:
        raise Exception()



def get_filtered_pockets(pockets:List[Pocket], k=5) -> List[Pocket]:
    def evaluate(pocket:Pocket):
        return pocket.drug_score + 0.1 * pocket.pocket_score
    pockets = sorted(pockets, key=evaluate, reverse=True)
    if len(pockets) > k:
        pockets = pockets[:k]
    return pockets

def get_neighbor_matrix(chain:Chain, pockets:List[Pocket], margin=2.0):
    """
    atm_coord: (num_res, max_atm_per_res, 3)
    atm_mask: (num_res, max_atm_per_res)
    for each pocket:
        vtx_coord: (num_vtx, 3)
        vtx_radius: (num_vtx)
    """
    neighb_matrix = np.array([[False for _ in range(len(pockets))] for _ in range(len(chain))])
    max_atm_per_res = max([len(res.atoms) for res in chain])
    atm_coord = np.zeros((len(chain), max_atm_per_res, 3))
    atm_mask = np.zeros((len(chain), max_atm_per_res), dtype=bool)
    for i, res in enumerate(chain):
        for j, atm in enumerate(res.atoms.values()):
            atm:Atom
            atm_coord[i, j, :] = atm.coord
            atm_mask[i, j] = True 
    for i, pocket in enumerate(pockets):
        vtx_coord = np.stack([vtx.center for vtx in pocket.verts], axis=0)
        vtx_radius = np.array([vtx.radius for vtx in pocket.verts])
        sqrdist = np.sum((atm_coord[:, :, None, :] - vtx_coord[None, None, :, :]) ** 2, axis=-1)
        sqrmargin = (vtx_radius[None, None, :] + margin) ** 2
        within_margin = (sqrdist < sqrmargin)
        atm_vtx_neighb = np.logical_and(within_margin, atm_mask[:, :, None])
        atm_neighb = np.any(atm_vtx_neighb, axis=-1)
        neighb = np.any(atm_neighb, axis=-1)
        assert neighb.shape == (len(chain),)
        neighb_matrix[:, i] = neighb
    return neighb_matrix

def save_d_list(dataset='kiba', skip_indices=[128], ver=1):
    l = []
    pdb_dir = Path(getattr(paths, f'{dataset}_pdb'))
    for i, (accession, seq) in enumerate(gen_name_seq_pairs(dataset)):
        if i in skip_indices:
            l.append(None)
            continue
        pdb_file = str(pdb_dir / f'{accession}.pdb')
        fpocket_dir = pdb_dir / f'{accession}_out'
        with open(pdb_file, 'r') as reader:
            chain = ProteinInfo.parse(reader.readlines).chains['A']
        pockets = parse_fpocket_output(fpocket_dir, skip_atoms=True)
        pockets = get_filtered_pockets(pockets, k=4)
        
        pocket_matrix = get_neighbor_matrix(chain, pockets, margin=2.0)

        R = np.float32(np.stack([get_rotation(res['N'].coord, res['CA'].coord, res['C'].coord) for res in chain], axis=0))
        t = np.float32(np.stack([res['CA'].coord for res in chain], axis=0))
        
        p = ''.join([convert_residue.get(res.resName, 'X') for res in chain])
        assert p == seq
        
        d = {
            'p': p,
            'R': torch.from_numpy(R),
            't': torch.from_numpy(t),
            'pocket_matrix': torch.from_numpy(pocket_matrix)
        }
        l.append(d)
    torch.save(l, str(Path(paths.dataset) / f'data' / f'{dataset}_d_list_ver{ver}.pth'))
    
if __name__ == '__main__':
    save_d_list(dataset='kiba', skip_indices=[128], ver=1)
