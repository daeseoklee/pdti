from typing import List, Tuple, Dict, Any, Union
from abc import ABCMeta, abstractmethod, abstractclassmethod

import torch
import torch.nn as nn 

from model.layers import FeedForward, LayerNorm, MaskedMaxpool, RayAttention
from asym.data_collection import DataCollection
from asym.annotated_module import AnnotatedModule
from asym.grouper import UniGrouper


def get_all_pocket_embedders():
    return {
        'rayatt1': RayAttentionModel1
    }

def get_pocket_embedder(config) -> 'PocketEmbedder':
    model_name = config['model']
    all_models = get_all_pocket_embedders()
    if model_name in all_models:
        model = all_models[model_name]
        return model(config)
    else:
        raise Exception(f'No such pocket embedder model - {model_name}')

class PocketEmbedder(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()

    def load_pretrained(self, config):
        checkpoint_path = config['pretrained']
        if checkpoint_path is None:
            return
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        refined_state_dict = {}
        for name, _ in self.named_parameters():
            if not f'model.{name}' in state_dict:
                raise Exception(f'The model to load pretrained state_dict from has no parameter {name}')
            refined_state_dict[name] = state_dict[f'model.{name}']
        self.load_state_dict(refined_state_dict)
    @abstractmethod
    def get_dim_embedding(self):
        pass
    

class RayAttentionModel1(PocketEmbedder):
    def __init__(self, config):
        super().__init__(config)
        model_config = config['model_config']
        #self.pocket_grouper = LengthThresholdGrouper(config['pocket_group_thresholds'])
        self.initial_transform = FeedForward(model_config['emb_dim'], model_config['initial_ff']['layers'], dropout_p=model_config['initial_ff']['dropout_p'])
        hidden_dim = self.initial_transform.last_dim
        self.layernorm_before_RA = LayerNorm(hidden_dim)
        if model_config['RA']['num_layers'] > 0:
            self.RA = RayAttention(hidden_dim=self.initial_transform.last_dim, **model_config['RA'])
        else:
            assert model_config['RA']['num_layers'] == 0
        if 'ff_before_maxpool' in model_config:
            this_config = model_config['ff_before_maxpool']
            self.ff_before_maxpool = FeedForward(hidden_dim, this_config['layers'], dropout_p=this_config['dropout_p'])
            self.last_dim = self.ff_before_maxpool.last_dim
        else:
            self.last_dim = hidden_dim
        self.maxpool = MaskedMaxpool(-2)

        self.load_pretrained(config)
            
    def get_dim_embedding(self):
        return self.last_dim
        
    def forward(self, pocket_dc:DataCollection):
#Turn this into a DataCollection method
        #dc = dc.group(self.pocket_grouper)
        pocket_dc['x'] = pocket_dc['x'].apply(self.initial_transform) 
        pocket_dc['x'] = pocket_dc['x'].apply(self.layernorm_before_RA) 
        if hasattr(self, 'RA'):
            x = pocket_dc.apply(self.RA, mask_hint=['pock'])
        else:
            x = pocket_dc['x']
        if hasattr(self, 'ff_before_maxpool'):
            x = x.apply(self.ff_before_maxpool)
        x = x.apply(self.maxpool)
        return x