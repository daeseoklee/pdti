from asym.precompute_grouper_thresholds import find_thresholds
from paths import paths 
from pathlib import Path 
import pickle, json

def compute_and_save(dataset='kiba', data_ver=1, num_thresholds=3):
    d_list_file = Path(paths.dataset) / 'data' / f'{dataset}_d_list_ver{data_ver}.pkl'
    with open(d_list_file, 'rb') as reader:
        l = pickle.load(reader)
    sizes = []
    for d in l:
        if d is None:
            continue
        pocket_matrix = d['pocket_matrix']
        for i in range(pocket_matrix.shape[1]):
            size = pocket_matrix[:, i].sum()
        sizes.append(size)
    square_fn = lambda x: x ** 2
    thresholds = find_thresholds(sizes, square_fn, num_thresholds, num_trials=1000, print_status=True)
    thresholds = [int(num) for num in thresholds]
    print('thresholds:', thresholds)
    threshold_file = str(Path(paths.pocket_thresholds) / f'{dataset}_ver{data_ver}_{num_thresholds}.json')
    with open(threshold_file, 'w') as writer:
        json.dump(thresholds, writer)
        
if __name__ == '__main__':
    compute_and_save(dataset='kiba', data_ver=1, num_thresholds=6)