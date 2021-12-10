from pathlib import Path
from paths import paths
from typing import List, Tuple, Dict, Any, Union
import pickle, json
import random
from vocab import SM_VOCAB, PR_VOCAB

import numpy as np
import torch
from torch.utils.data import Dataset

from asym.data_collection import DataCollection
from asym.grouper import LengthThresholdGrouper, PredefinedConsecutiveGrouper
from asym.padding import CDimPadder


def load_file(file):
    filename = str(file)
    if filename.endswith('.json'):
        with open(filename, 'r') as reader:
            return json.load(reader)
    elif filename.endswith('.txt'):
        with open(filename, 'r') as reader:
            return json.load(reader)
    elif filename.endswith('.pkl'):
        with open(filename, 'rb') as reader:
            return pickle.load(reader)
    with open(filename, 'rb') as reader:
        return pickle.load(reader, encoding='latin1')

class PdtiDataset(Dataset):
    def __init__(self, config):
        dataset = config['dataset'] #kiba, davis etc.
        mode = config['mode'] #cv or test
        split = config['split'] #trn or val
        fold = config['fold'] #0~4
        unseen = config['unseen'] #True or False
        pocket_ver = config['pocket_ver'] #1
        pemb_model = config['pemb_model'] #prot_bert
        num_thresholds= config['num_thresholds']
        
        
        self.Y, self.m_idxs, self.p_idxs = self.get_deepdta_data(dataset)
        self.data_idxs = self.get_data_idxs(dataset, mode, fold, split, unseen)
        
        self.pockets = self.get_pockets(dataset, pemb_model, ver=pocket_ver)
        
        self.molecule_max_len = 100
        self.molecules = self.get_molecules(dataset)
        
        self.protein_max_len = 1000
        self.proteins = self.get_proteins(dataset)
        
        self.pocket_thresholds = self.get_pocket_thresholds(dataset, pocket_ver=pocket_ver, num_thresholds=num_thresholds)
        
    def get_deepdta_data(self, dataset):
        if not dataset in ['kiba', 'davis']:
            raise Exception('Not implemented yet')
        base_dir = Path(paths.deepdta) / 'data' / dataset
        Y = torch.from_numpy(np.float32(load_file(base_dir / 'Y')))
        m_idxs, p_idxs = torch.where(~torch.isnan(Y))
        
        return Y, m_idxs, p_idxs
        
    
    def get_data_idxs(self, dataset, mode, fold, split, unseen):
        if not dataset in ['kiba', 'davis']:
            raise Exception('Not implemented yet')
        base_dir = Path(paths.deepdta) / 'data' / dataset / 'folds'
        
        if mode == 'cv' or (mode == 'test' and split == 'trn'):
            if unseen:
                file = base_dir / 'train_fold_setting3.txt'
            else:
                file = base_dir / 'train_fold_setting2.txt'
            train_folds = load_file(file)
            assert len(train_folds) == 5
            if split == 'trn':
                trn_set = []
                for i, l in enumerate(train_folds):
                    if i != fold:
                        trn_set += l
                return trn_set
            elif split == 'val':
                val_set = train_folds[fold]
                return val_set
            else:
                raise Exception("Can't happen")
        elif mode == 'test' and split == 'val':
            if unseen:
                file = base_dir / 'test_fold_setting3.txt'
            else:
                file = base_dir / 'test_fold_setting2.txt'
            val_set = load_file(file)
            return val_set
            
        else:
            raise Exception("Can't happen")
    
    def get_pockets(self, dataset, pemb_model, ver=1):
        file = Path(paths.precomputed_pockets) / f'{dataset}_ver{ver}_{pemb_model}.pth'
        return torch.load(file)
    
    def prefetch_pockets(self, device):
        for l in self.pockets:
            if l is None:
                continue
            for pocket in l:
                pocket['x'] = pocket['x'].to(device)
                pocket['R'] = pocket['R'].to(device)
                pocket['t'] = pocket['t'].to(device)
    
    def get_precomputed_pockets(self, dataset, pemb_model, ver=1):
        file = Path(paths.precomputed_pockets) / f'{dataset}_ver{ver}_{pemb_model}.pth'
        return torch.load(file)
    
    def get_proteins(self, dataset):
        file = Path(paths.deepdta) / 'data' / dataset / 'proteins.txt'
        return list(load_file(file).values())
    
    def get_molecules(self, dataset):
        file = Path(paths.deepdta) / 'data' / dataset / 'ligands_iso.txt'
        return list(load_file(file).values())
    
    def get_pocket_thresholds(self, dataset, pocket_ver=1, num_thresholds=2):
        file = Path(paths.pocket_thresholds) / f'{dataset}_ver{pocket_ver}_{num_thresholds}.json'
        return load_file(file)

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, i):
        idx = self.data_idxs[i]
        m_idx = self.m_idxs[idx]
        p_idx = self.p_idxs[idx]
        
        
        affinity = self.Y[m_idx, p_idx]
        m = self.get_m(m_idx)
        p = self.get_p(p_idx)
        pockets = self.pockets[p_idx]
                
        return {
            'pockets': pockets,
            'p': p, 
            'm': m,
            'affinity': affinity
        }
        
    def collate_fn(self, batch):
        flattened_pockets = []
        for d in batch:
            for pocket in d['pockets']:
                flattened_pockets.append(pocket)
        pocket_shapesig = {
            'x': '(B, L_pock, M_emb)',
            'R': '(B, L_pock, 3, 3)',
            't': '(B, L_pock, 3)'
        }
        pocket_grouper = LengthThresholdGrouper('pock', self.pocket_thresholds)
        identity_matrix = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        padding = {
            'R': CDimPadder(identity_matrix) 
        }
        pocket_dc = DataCollection(pocket_shapesig, data_list=flattened_pockets)
        pocket_dc.group(pocket_grouper, padding=padding)
        
        nums = [len(d['pockets']) for d in batch]
        pocket_regrouper = PredefinedConsecutiveGrouper(nums)
        
        p = [d['p'] for d in batch]
        p_dc = DataCollection('(B, L_res)', data_list=p)
        
        m = [d['m'] for d in batch]
        m_dc = DataCollection('(B, L_atm)', data_list=m)
        
        affinity = torch.tensor([d['affinity'] for d in batch])
        
        return (pocket_dc, pocket_regrouper, p_dc, m_dc, affinity)
        
    def get_crop_offset(self, m, max_len):
        if type(m) == list or type(m) == str:
            m_len = len(m)
        elif type(m) == torch.Tensor:
            m_len = m.shape[0]
        else:
            raise Exception()
        
        if m_len <= max_len:
            return None
        return random.randint(0, m_len - max_len)

    def get_m(self, m_idx):
        
        m = self.molecules[m_idx]
        offset = self.get_crop_offset(m, self.molecule_max_len)
        if offset is not None:
            m = m[offset : offset + self.molecule_max_len]
        m = [SM_VOCAB[c] for c in m]

        m = [SM_VOCAB['[CLS]'], SM_VOCAB['[BEGIN]']] + m + [SM_VOCAB['[END]']] 

        return torch.tensor(m)
        

    def get_p(self, p_idx):
        """
        return a dictionary with 
        keys 
        -'p', 
        and optional keys
        -'p_mask', 'p_content_mask', 'p_content_range', 'p_R', 'p_t'
        """

        p = self.proteins[p_idx]
    
        offset = self.get_crop_offset(p, self.protein_max_len)

        if offset is not None:
            p = p[offset : offset + self.protein_max_len]
            
        p = [PR_VOCAB.get(c, PR_VOCAB['X']) for c in p]

        return torch.tensor(p)



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset_config = {
        'dataset': 'kiba',
        'mode': 'cv',
        'split': 'trn',
        'fold': 0,
        'unseen': True,
        'pemb_model': 'prot_bert'
    }
    dataset = PdtiDataset(dataset_config)
    loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
    dc, regrouper, p, m, aff = next(loader.__iter__()) 
    print('here')
        
        

    