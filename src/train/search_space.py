from argparse import Namespace
from copy import deepcopy
from typing import List, Dict, Tuple, Union, Any, Generator, Callable
from itertools import product
from math import log10, floor


def format_val(val):
    if isinstance(val, str):
        return val
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, float):
        if len(str(val)) <= 5:
            return str(val)
        return '{:.1e}'.format(val)

def iter_search_space(args: Namespace, search_fn: Callable[[], Generator[Dict[str, Tuple[Union[str, int, float], bool]], None, None]]):
    """
    generate modified [args] iterated or sampled according to [search_fn()] generator outputs
    """
    arg_to_short = {
        'lr': 'l',
        'scheduler': 's',
        'batch': 'b',
        'msa': 'c',  #coverage
        'msa_mult': 'm',
        'iso_all': 'iall' 
    }
    
    for v, d in search_fn():
        new_args = deepcopy(args)
        new_args.search = None
        
        version = v 
        
        for arg, (val, append_to_version) in d.items():
            setattr(new_args, arg, val)
            if append_to_version:
                short = arg_to_short[arg]
                if val == isinstance(val, bool):
                    if val:
                        version += f'_{short}'
                    else:
                        pass
                else:
                    version += f'_{short}:{format_val(val)}'
        if version == '':
            raise Exception('Version name cannot be an empty string')
        if version.startswith('_'):
            version = version[1:]
        new_args.version = version
        
        yield new_args

#------------------------------------------------------

"""
nohup python src/train/dti_train.py --dataset kiba --challenge --mode cv --fold 1 --search search_20210817 --experiment 210817_msa_config --gpus 8 --num_workers 16 --batch 64 --disable_weight_record & 
"""
        
def search_20210817():
    common = {
        'model': ('simple', False), 
        'pemb': ('cnn_point', False),
        'memb': ('bert_point', False),
        'epochs': (1000, False),
        'warmup': (10000, False),
        'lr': (1e-4, True),
        'scheduler': ('lin', True)
    }
    

    for msa, msa_mult in [(60, 0.5), (60, 1.0), (60, 2.0), (70, 0.5), (70, 1.0), (70, 2.0), (80, 0.5), (80, 1.0), (80, 2.0)]:

        d = deepcopy(common)
        d['msa'] = (msa, True)
        d['msa_mult'] = (msa_mult, True)
        d['iso_all'] = (True, False)
 
        yield '', d
    
    d = deepcopy(common)
    d['iso_all'] = (False, False)
    yield '', d
    
    
"""
nohup python src/train/dti_train.py --dataset kiba --mode test --fold 0 --search mtdti_main_test --experiment mtdti_main_test --gpus 8 --num_workers 16 --batch 64 --disable_weight_record & 
"""

            
def mtdti_main_test():
    common = {
        'model': ('simple', False), 
        'pemb': ('cnn_point', False),
        'memb': ('bert_point', False),
        'epochs': (1000, False),
        'warmup': (10000, False),
        'lr': (1e-4, True),
        'scheduler': ('lin', True)
    }
    
    for challenge in [False, True]:

        d = deepcopy(common)
        d['challenge'] = (challenge, False)
        d['msa'] = (70, True)
        d['msa_mult'] = (0.5, True)
        d['iso_all'] = (True, False)
        yield '', d
        
        d = deepcopy(common)
        d['challenge'] = (challenge, False)
        d['iso_all'] = (False, False)
        yield '', d

    

                     
            
        
    
    
"""
python src/train/dti_train.py --dataset kiba --mode test --fold 1 --experiment length_revised --search length_revised_200830 --batch 64 --gpus 8 --num_workers 8 --disable_weight_record
"""

def length_revised_200830():
    common = {
        'model': ('simple', False), 
        'pemb': ('cnn_point', False),
        'epochs': (1200, False),
        'warmup': (10000, False),
        'lr': (1e-4, False),
        'scheduler': ('lin', False)
    }

    for v in ['mt', 'mat']:
        d = deepcopy(common)
        d['challenge'] = (False, False)
        if v == 'mt':
            d['memb'] = ('bert_point', False)
        elif v == 'mat':
            d['memb'] = ('mat_point', False)
        yield v, d 

    
    for v in ['mt', 'mt+msa', 'mt+msa+noiso','mt+msa+filter', 'mat']: #'mat+msa', 'mat+msa+noiso', 'mat+msa+filter'
        d = deepcopy(common)
        d['challenge'] = (True, False)
        if 'mt' in v:
            d['memb'] = ('bert_point', False)
        elif 'mat' in v:
            d['memb'] = ('mat_point', False)
        if 'msa' in v:
            d['msa'] = (70, False)
            d['msa_mult'] = (0.5, False)
            d['iso_all'] = (True, False)
            if 'filter' in v:
                d['msa_median_acc'] = (True, False)
                d['msa_median_acc_from'] = (850, False)
            if 'noiso' in v:
                d['iso_all'] = (False, False)
        yield v, d


"""
python src/train/dti_train.py --dataset kiba --mode test --fold 2 --experiment length_revised --search length_revised_200831 --batch 64 --gpus 8 --num_workers 8 --disable_weight_record
wait
python src/train/dti_train.py --dataset kiba --mode test --fold 3 --experiment length_revised --search length_revised_200831 --batch 64 --gpus 8 --num_workers 8 --disable_weight_record
"""

def length_revised_200831():
    common = {
        'model': ('simple', False), 
        'pemb': ('cnn_point', False),
        'epochs': (1200, False),
        'warmup': (10000, False),
        'lr': (1e-4, False),
        'scheduler': ('lin', False)
    }


    
    for v in ['mt', 'mt+msa','mt+msa+filter', 'mt+msa+noiso']: #'mat+msa', 'mat+msa+noiso', 'mat+msa+filter'
        d = deepcopy(common)
        d['challenge'] = (True, False)
        if 'mt' in v:
            d['memb'] = ('bert_point', False)
        elif 'mat' in v:
            d['memb'] = ('mat_point', False)
        if 'msa' in v:
            d['msa'] = (70, False)
            d['msa_mult'] = (0.5, False)
            d['iso_all'] = (True, False)
            if 'filter' in v:
                d['msa_median_acc'] = (True, False)
                d['msa_median_acc_from'] = (850, False)
            if 'noiso' in v:
                d['iso_all'] = (False, False)
        yield v, d