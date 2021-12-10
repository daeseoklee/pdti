from typing import Tuple, List, Dict, Any

from model.bert import Bert, load_pretrained_bert
from model.mat import load_pretrained_mat

from abc import ABCMeta, abstractmethod, abstractclassmethod

import torch
import torch.nn as nn

from asym.annotated_module import AnnotatedModule


def get_all_molecule_embedders():
    return {
        'bert': BertMoleculeEmbedder,
        'mat': MATMoleculeEmbedder
    }

def get_molecule_embedder(memb_config:Dict[str, Any]) -> 'MoleculeEmbedder':
    model_name = memb_config['model']
    all_memb_models = get_all_molecule_embedders()
    if model_name in all_memb_models:
        memb_model = all_memb_models[model_name](memb_config)
    else:
        raise Exception(f'molecule embedder "{model_name}" not implemented')
    return memb_model

def get_molecule_input_format(model_name:str) -> str:
    all_memb_models = get_all_molecule_embedders()
    if model_name in all_memb_models:
        memb_model_class:MoleculeEmbedder = all_memb_models[model_name]
    else:
        raise Exception(f'molecule embedder "{model_name}" not implemented')
    molecule_input_format = memb_model_class.get_molecule_input_format()
    return molecule_input_format

class MoleculeEmbedder(AnnotatedModule, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
    @abstractmethod
    def get_dim_embedding(self):
        pass
    @abstractclassmethod
    def get_molecule_input_format(cls):
        pass
    @abstractmethod
    def forward(self, m:Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
    

    
class BertMoleculeEmbedder(MoleculeEmbedder):
    def __init__(self, config):
        super().__init__(config)
        self.bert:Bert = load_pretrained_bert(config['model_config'], config['pretrained'], map_location=torch.device('cpu'))
    
    @classmethod
    def get_molecule_input_format(cls):
        return 'bert'
    
    def forward(self, m, mask):
        return self.bert(m, attention_mask=mask)[:, 0, :]

    def get_mask_hint(self):
        return 'copy'
    
    def get_input_annot(self):
        return '(b, l_atm)'
    
    def get_output_annot(self):
        return '(b, m_molecule_feature)'

    def get_dim_embedding(self):
        return self.bert.dim_hidden 

class MATMoleculeEmbedder(MoleculeEmbedder): # This should become AnnotatedModule
    def __init__(self, config):
        super().__init__(config)
        self.mat = load_pretrained_mat(config['model_config'], config['pretrained'])
        
    @classmethod
    def get_molecule_input_format(cls):
        return 'mat'
    
    def forward(self, m):
        node_feature = m['node_feature']
        adj_matrix = m['adj_matrix']
        dist_matrix = m['dist_matrix']
        out = self.mat(node_feature, adj_matrix, dist_matrix)
        return out
    
    def get_dim_embedding(self):
        return self.mat.generator.n_output



    