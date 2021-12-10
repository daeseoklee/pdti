from typing import Tuple, List, Dict, Union, Any
import numpy as np
import json, pickle
from random import randint

from paths import paths

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

def get_seen_folds(m_idxs, p_idxs, orig_train_folds, orig_test_fold, skip_indices=[128]):
    def is_ok(i):
        return not p_idxs[i] in skip_indices
    new_train_folds = [list(filter(is_ok, fold)) for fold in orig_train_folds]
    new_test_fold = list(filter(is_ok, orig_test_fold))
    return new_train_folds, new_test_fold

def get_unseen_folds(m_idxs, p_idxs, k=5, skip_indices=[128]):
    assert len(m_idxs) == len(p_idxs)
    n = len(m_idxs)
    l = [i for i in range(n) if not p_idxs[i] in skip_indices]
    p_map = {}
    for i in l:
        p_idx = p_idxs[i]
        if not p_idx in p_map:
            p_map[p_idx] = []
        p_map[p_idx].append(i)
    partition = [[] for _ in range(k+1)]
    for sub_l in p_map.values():
        j = randint(0, k)
        partition[j] += sub_l
    for sub_l in partition:
        sub_l.sort()
    new_train_folds = partition[:k]
    new_test_fold = partition[-1]

    return new_train_folds, new_test_fold
    

if __name__ == '__main__':
    skip_indices = [128]
    exclude = []
    prefix = paths.deepdta + '/data/kiba/'
    Y = load_file(prefix + 'Y')
    m_idxs, p_idxs = np.where(~np.isnan(Y))
    train_folds = load_file(prefix + 'folds/train_fold_setting1.txt')
    test_fold = load_file(prefix + 'folds/test_fold_setting1.txt')
    
    seen_train_folds, seen_test_fold = get_seen_folds(m_idxs, p_idxs, train_folds, test_fold, skip_indices=[128])
    unseen_train_folds, unseen_test_fold = get_unseen_folds(m_idxs, p_idxs, k=5, skip_indices=[128])

    with open(prefix + 'folds/train_fold_setting2.txt', 'w') as writer:
        json.dump(seen_train_folds, writer)

    with open(prefix + 'folds/test_fold_setting2.txt', 'w') as writer:
        json.dump(seen_test_fold, writer)
        
    with open(prefix + 'folds/train_fold_setting3.txt', 'w') as writer:
        json.dump(unseen_train_folds, writer)

    with open(prefix + 'folds/test_fold_setting3.txt', 'w') as writer:
        json.dump(unseen_test_fold, writer)

    print(sum([len(fold) for fold in train_folds]) + len(test_fold))
    print(sum([len(fold) for fold in seen_train_folds]) + len(seen_test_fold))
    print(sum([len(fold) for fold in unseen_train_folds]) + len(unseen_test_fold))

    print('finished!')
    