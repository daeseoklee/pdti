from vocab import PR_VOCAB
from model.bert import Bert, load_pretrained_bert

from typing import Tuple, List, Dict, Any
from collections import namedtuple
from abc import ABCMeta, abstractmethod, abstractclassmethod

import torch.nn as nn 
import torch
from torch.nn.functional import pad

from transformers import BertModel

from asym.annotated_module import AnnotatedModule

def get_all_protein_embedders():
    return {
        'cnn': CnnProteinEmbedder,
        'bert': BertProteinEmbedder,
        'prot_bert': ProtBertPointEmbedder,
        'precomputed_prot_bert': PrecomputedProtBertPointEmbedder,
        'precomputed_prot_t5_xl': PrecomputedProtT5XLPointEmbedder,
    }

def get_protein_embedder(config:Dict[str, Any]) -> 'ProteinEmbedder':
    model_name = config['model']
    all_models = get_all_protein_embedders()
    if model_name in all_models:
        pemb_model = all_models[model_name](config)
    else:
        raise Exception(f'protein embedder "{model_name}" not implemented')
    return pemb_model


def get_protein_input_format(model_name:str) -> str:
    all_models = get_all_protein_embedders()
    if model_name in all_models:
        pemb_model_class:ProteinEmbedder = all_models[model_name]
    else:
        raise Exception(f'protein embedder "{model_name}" not implemented')
    protein_input_format = pemb_model_class.get_protein_input_format()
    return protein_input_format
        

class ProteinEmbedder(AnnotatedModule, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
    @abstractmethod
    def get_dim_embedding(self):
        pass
    @abstractclassmethod
    def get_protein_input_format(cls):
        pass
    @abstractmethod
    def forward(self, p:Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    

    
class BertProteinEmbedder(ProteinEmbedder):
    def __init__(self, config):
        super().__init__(config)
        model_config = config['model_config']
        pretrained = config['pretrained']
        self.bert:Bert = load_pretrained_bert(model_config, pretrained)
    
    @classmethod
    def get_protein_input_format(cls):
        return 'bert'

    def forward(self, p, mask):
        return self.bert(p, attention_mask=mask)[:, 0, :]
    
    def get_dim_embedding(self):
        return self.bert.dim_hidden 
    

    
class ProtBertPointEmbedder(ProteinEmbedder):# This should become AnnotatedModule 
    def __init__(self, config):
        super().__init__(config)
        model_config = config['model_config']
        pretrained = config['pretrained']
        self.bert = BertModel.from_pretrained(pretrained)
    def get_dim_embedding(self):
        return 1024
    @classmethod
    def get_protein_input_format(cls):
        return 'prot_bert'

    def forward(self, p):
        p = p['p']
        p_mask = p['p_mask']
        return self.bert(p, attention_mask=p_mask).last_hidden_state[:, 0, :]


class PrecomputedProtBertPointEmbedder(ProteinEmbedder): # This should become AnnotatedModule 
    def __init__(self, config):
        super().__init__(config)
        model_config = config['model_config']
        self.dim_input = model_config['dim_input']
        self.dim_output = model_config['dim_output']
        dropout_p = model_config['dropout_p']
        self.linear = nn.Linear(self.dim_input, self.dim_output)
        self.activation = nn.GELU() 
        self.dropout = nn.Dropout(p=dropout_p)
        self.layernorm = nn.LayerNorm(self.dim_output)
    def get_dim_embedding(self):
        return self.dim_output
    @classmethod
    def get_protein_input_format(cls):
        return 'precomputed_prot_bert'

    def forward(self, p):
        p = p['p']
        p = self.activation(self.linear(p))
        p = self.dropout(p)
        p = self.layernorm(p)
        return p

    

class PrecomputedProtT5XLPointEmbedder(ProteinEmbedder):# This should become AnnotatedModule 
    def __init__(self, config):
        super().__init__(config)
        model_config = config['model_config']
        self.dim_input = model_config['dim_input']
        self.dim_output = model_config['dim_output']
        dropout_p = model_config['dropout_p']
        self.linear = nn.Linear(self.dim_input, self.dim_output)
        self.activation = nn.GELU() 
        self.dropout = nn.Dropout(p=dropout_p)
        self.layernorm = nn.LayerNorm(self.dim_output)
    def get_dim_embedding(self):
        return self.dim_output
    @classmethod
    def get_protein_input_format(cls):
        return 'precomputed_prot_t5_xl'

    def forward(self, p):
        p = p['p']
        p = self.activation(self.linear(p))
        p = self.dropout(p)
        p = self.layernorm(p)
        return p


        

class CnnProteinEmbedder(ProteinEmbedder):
    def __init__(self, config):
        super().__init__(config)
        print('----------------------------')
        #print(config.keys())
        model_config = config['model_config']
        first_channel_dim = model_config['channels'][0] 
        self.dim_embedding = model_config['channels'][-1]
        self.embed = nn.Embedding(len(PR_VOCAB), first_channel_dim)
        self.cnn = Cnn(**model_config)
    @classmethod
    def get_protein_input_format(cls):
        return 'raw'

    def forward(self, p:torch.Tensor):
        p = self.embed(p) #(batch, length, embedding)
        p = self.cnn(p.transpose(1, 2)) #(batch, last_channel, length)
        p = torch.max(p, dim=2).values
        return p
    
    def get_mask_hint(self):
        return None
    
    def get_input_annot(self):
        return '(b, l_res)'
    
    def get_output_annot(self):
        return '(b, m_protein_feature)'
    
    def get_dim_embedding(self):
        return self.dim_embedding


    

class Cnn(nn.Module):
    def __init__(self, max_seq_len=1000, channels=[32, 32, 64, 96], kernels=[12, 12, 12], dropout_p=0.1, pad_same=False):
        super().__init__()
        assert len(channels) == len(kernels) + 1
        self.pad_same = pad_same
        self.kernels = kernels
        convolutions = []
        for i in range(len(kernels)):
            convolutions.append(nn.Conv1d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=kernels[i], stride=1, padding=0, bias=True))
        self.convolutions = nn.ModuleList(convolutions)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self, x):
        for i, convolution in enumerate(self.convolutions):
            if self.pad_same:
                x = pad(x, (0, self.kernels[i] - 1))
            x = self.activation(convolution(x))
            x = self.dropout(x)
        return x
    
    
if __name__ == '__main__':
    pass