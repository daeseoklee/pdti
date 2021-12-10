from paths import paths
from pathlib import Path
import torch


def precompute_and_save(dataset, emb_model, ver=1):
    emb_file = Path(paths.precomputed_embeddings) / f'{dataset}_{emb_model}.pth'
    d_list_file = Path(paths.dataset) / f'data' / f'{dataset}_d_list_ver{ver}.pth'
    
    xs = torch.load(emb_file)
    d_list = torch.load(d_list_file)
    
    l = []
    for x, d in zip(xs, d_list):
        assert len(x.shape) == 2
        if d is None:
            l.append(None)
            continue 
        R = d['R']
        t = d['t']
        pocket_matrix = d['pocket_matrix']
        assert x.shape[0] == R.shape[0] == t.shape[0] == pocket_matrix.shape[0]
        
        pockets = []
        for j in range(pocket_matrix.shape[1]):
            in_pock = pocket_matrix[:, j]
            pock_x = x[in_pock, :]
            pock_R = R[in_pock, :, :]
            pock_t = t[in_pock, :]
            pockets.append({
                'x': pock_x,
                'R': pock_R,
                't': pock_t
            })
        l.append(pockets)
        
    file = Path(paths.precomputed_pockets) / f'{dataset}_ver{ver}_{emb_model}.pth'
    
    torch.save(l, file)
        

if __name__ == '__main__':
    precompute_and_save('kiba', 'prot_bert', ver=1)
        
    