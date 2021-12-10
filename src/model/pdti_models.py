from typing import List, Tuple, Dict, Any, Union

import torch
from torch import Tensor
import torch.nn as nn 

from model.layers import FeedForward, LayerNorm, MaskedMaxpool, RayAttention, MultiHeadRegressor
from asym.data_collection import DataCollection
from asym.annotated_module import AnnotatedModule
from asym.grouper import UniGrouper, DataListGrouper

from model.pocket_embedders import get_pocket_embedder
from model.protein_embedders import get_protein_embedder
from model.molecule_embedders import get_molecule_embedder

import time


class PocketBasedRegressor(AnnotatedModule):
    """ 
    Which information should be given additio
    """
    def __init__(self, pocket_feature_dim, molecule_feature_dim, config):
        super().__init__()
        self.pocket_feature_dim = pocket_feature_dim
        self.molecule_feature_dim = molecule_feature_dim
        concat_dim = pocket_feature_dim + molecule_feature_dim 
        self.hidden_layers = FeedForward(concat_dim, config['layers'], dropout_p=config['dropout_p'])
        self.heads = MultiHeadRegressor(config['layers'][-1], ['pocket_affinity_scores', 'pocket_confidence_scores'], zero_weight=['pocket_affinity_scores', 'pocket_confidence_scores'])
        
    def forward(self, d:Dict[str, Tensor]):
        pocket_features = d['pocket_features']
        molecule_feature = d['molecule_feature']
        N, L_pockets, _ = pocket_features.shape
        assert molecule_feature.shape == (N, self.molecule_feature_dim)
        broadcasted_molecule_features = molecule_feature[:, None, :].broadcast_to((N, L_pockets, self.molecule_feature_dim))
        
        x = torch.cat([pocket_features, broadcasted_molecule_features], dim=2)
        x = self.hidden_layers(x)
        return self.heads(x)
    def get_mask_hint(self):
        return None
    def get_input_annot(self) -> Union[str, Dict[str, Any]]:
        return {
            'pocket_features': '(b, l_pockets, m_pocket)',
            'molecule_feature': '(b, m_molecule)'
        }
    def get_output_annot(self) -> Union[str, Dict[str, Any]]:
        return {
            'pocket_affinity_scores': '(b, l_pockets)',
            'pocket_confidence_scores': '(b, l_pockets)'
        }

class ProteinBasedRegressor(AnnotatedModule):
    def __init__(self, protein_feature_dim, molecule_feature_dim, config):
        super().__init__()
        self.protein_feature_dim = protein_feature_dim
        self.molecule_feature_dim = molecule_feature_dim
        concat_dim = protein_feature_dim + molecule_feature_dim 
        self.hidden_layers = FeedForward(concat_dim, config['layers'], dropout_p=config['dropout_p'])
        self.head = nn.Linear(config['layers'][-1], 1)
        self.weight_init_()
    def weight_init_(self):
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight) 
    def forward(self, d:Dict[str, Tensor]):
        protein_feature = d['protein_feature']
        molecule_feature = d['molecule_feature']
        N = protein_feature.shape[0]
        assert protein_feature.shape == (N, self.protein_feature_dim)
        assert molecule_feature.shape == (N, self.molecule_feature_dim)
        
        x = torch.cat([protein_feature, molecule_feature], dim=1)
        x = self.hidden_layers(x)
        return self.head(x)
    
    def get_mask_hint(self):
        return None
    def get_input_annot(self) -> Union[str, Dict[str, Any]]:
        return {
            'protein_feature': '(b, m_protein)',
            'molecule_feature': '(b, m_molecule)'
        }
    def get_output_annot(self) -> Union[str, Dict[str, Any]]:
        return '(b, 1)'


class ScoreAggregator(AnnotatedModule):
    def __init__(self, use_whole=True):
        super().__init__()
        self.use_whole = use_whole
        self.softmax = nn.Softmax(-1)
        
    def forward(self, d:Dict[str, Tensor], mask:Tensor):
        N, L = mask.shape
        pocket_affinity_scores = d['pocket_affinity_scores']
        pocket_confidence_scores = d['pocket_confidence_scores']
        assert pocket_affinity_scores.shape == (N, L)
        assert pocket_confidence_scores.shape == (N, L)

        if self.use_whole:
            base_affinity_score = d['base_affinity_score']
            assert base_affinity_score.shape == (N, 1)
        
        negative_infinity = - torch.tensor(float('inf'), device=mask.device)
        weight = torch.where(mask, pocket_confidence_scores, negative_infinity) #(N, L)
        if self.use_whole:
            weight = torch.cat([weight, torch.zeros((N, 1), device=mask.device)], dim=1) #(N, L + 1)
        weight = self.softmax(weight) #(N, L) or (N, L + 1)
        
        if self.use_whole:
            scores = torch.cat([pocket_affinity_scores, base_affinity_score], dim=1) #(N, L + 1)
        else:
            scores = pocket_affinity_scores #(N, L)
        
        score =  torch.sum(weight * scores, dim=1) #(N, )
        return score
    
    def get_mask_hint(self):
        return ['pockets']
    def get_input_annot(self) -> Union[str, Dict[str, Any]]:
        d = {
            'pocket_affinity_scores': '(b, l_pockets)',
            'pocket_confidence_scores': '(b, l_pockets)', 
        }
        if self.use_whole:
            d['base_affinity_score'] = '(b, 1)'
        return d
    def get_output_annot(self) -> Union[str, Dict[str, Any]]:
        return '(b)'

class PdtiModel(nn.Module): 
    def __init__(self, model_config):
        super().__init__()
        pocket_embedder_config = model_config['pocket_emb']
        protein_embedder_config = model_config['protein_emb']
        if protein_embedder_config is None:
            self.use_whole = False
        else:
            self.use_whole = True
        molecule_embedder_config = model_config['molecule_emb']
        self.pocket_embedder = get_pocket_embedder(pocket_embedder_config)
        pocket_feature_dim = self.pocket_embedder.get_dim_embedding()
        if self.use_whole:
            self.protein_embedder = get_protein_embedder(protein_embedder_config)
            protein_feature_dim = self.protein_embedder.get_dim_embedding()
        self.molecule_embedder = get_molecule_embedder(molecule_embedder_config)
        molecule_feature_dim = self.molecule_embedder.get_dim_embedding()
        self.pocket_based_regressor = PocketBasedRegressor(pocket_feature_dim, molecule_feature_dim, model_config['pocket_reg'])
        if self.use_whole:
            self.protein_based_regressor = ProteinBasedRegressor(protein_feature_dim, molecule_feature_dim, model_config['protein_reg'])
        self.score_aggregator = ScoreAggregator(use_whole=self.use_whole)
        
    def forward(self, pocket_dc:DataCollection, pocket_regrouper:DataListGrouper, p:DataCollection, m:DataCollection, times=None, activations_to_record=[]):
        assert pocket_dc.is_grouped 
        assert not p.is_grouped
        assert not m.is_grouped
        
        activations = {}
        if times is not None:
            a = time.time()
        pocket_features = self.pocket_embedder(pocket_dc)
        if times is not None:
            b=time.time()
        pocket_features.regroup(pocket_regrouper)
        if times is not None:
            c=time.time()
        pocket_features = pocket_features.groups_as_data(0, 'pockets')
        if times is not None:
            d=time.time()
        pocket_features.group(UniGrouper())
        
        m.group(UniGrouper())
        p.group(UniGrouper())
        if times is not None:
            e=time.time()
        
        molecule_feature = m.apply(self.molecule_embedder)
        if self.use_whole:
            protein_feature = p.apply(self.protein_embedder)
        if times is not None:
            f=time.time()
        
        pocket_dc = DataCollection.from_dict({
            'pocket_features': pocket_features,
            'molecule_feature': molecule_feature
        })
        
        d = {'molecule_feature': molecule_feature}
        if self.use_whole:
            d['protein_feature'] = protein_feature
        protein_dc  = DataCollection.from_dict(d)
        if times is not None:
            g=time.time()
        
        intermediate_scores = pocket_dc.apply(self.pocket_based_regressor) #keys: 'pocket_affinity_scores', 'pocket_confidences_scores' 
        if self.use_whole:
            intermediate_scores['base_affinity_score'] = protein_dc.apply(self.protein_based_regressor) 

        if 'pocket_confidence_scores' in activations_to_record:
            pcs = intermediate_scores['pocket_confidence_scores']
            score = pcs.data_groups[0].value
            score_mask = pcs.get_mask(0, mask_hint='copy')
            mean_confidence_score = torch.sum(score * score_mask, dim=1) / torch.sum(score_mask, dim=1)
            inf = torch.tensor(float('inf'), device=score.device)
            max_confidence_score = torch.max(torch.where(score_mask, score, -inf), dim=1).values
            min_confidence_score = torch.min(torch.where(score_mask, score, inf), dim=1).values

            activations['mean_confidence_score'] = mean_confidence_score
            activations['max_confidence_score'] = max_confidence_score
            activations['min_confidence_score'] = min_confidence_score
        
        final_score = intermediate_scores.apply(self.score_aggregator)
        h=time.time()
        
        if times is not None:
            d = {
                'pocket_feature_extraction': b-a,
                'initial_regrouping': c-b,
                'group_as_data': d-c,
                'unigroupers': e-d,
                'pm_features': f-e,
                'from_dict':g-f,
                'final_score_calc':h-g
            }
            for key, val in d.items():
                if not key in times:
                    times[key] = 0
                times[key] += val

        return final_score.data_groups[0].value, activations #1D tensor of predicted affinity values, activation tensors
    

        
if __name__ == '__main__':
    from train.pdti_dataset import PdtiDataset
    from torch.utils.data import DataLoader
    from paths import paths
    from pathlib import Path
    import json
    
    dataset_config = {
        'dataset': 'kiba',
        'mode': 'cv',
        'split': 'trn',
        'fold': 0,
        'unseen': True,
        'pemb_model': 'prot_bert',
        'pocket_ver': 1,
        'num_thresholds': 3
    }
    
    pretrained = str(Path(paths.sample_pretrained_models) / 'rayatt1' / 'last.ckpt')
    memb_pretrained = str(Path(paths.sample_pretrained_models) / 'memb' / 'last.ckpt')
    
    with open(Path(paths.sample_pretrained_models) / 'rayatt1' / 'model_config.json', 'r') as reader:
        psp_model_config = json.load(reader)
    with open(Path(paths.model_config) / 'protein_embedders' / 'cnn.json') as reader:
        pemb_config = json.load(reader)
    with open(Path(paths.model_config) / 'molecule_embedders' / 'bert.json') as reader:
        memb_config = json.load(reader)
    model_config = {
        'pocket_emb': {
            'model': 'rayatt1',
            'pretrained': pretrained,
            'model_config': psp_model_config
        },
        'protein_emb': {
            'model': 'cnn',
            'model_config': pemb_config
        },
        'molecule_emb': {
            'model': 'bert',
            'model_config': memb_config,
            'pretrained': memb_pretrained
        },
        'pocket_reg': {
            'layers': [32],
            'dropout_p': 0.2
        },
        'protein_reg': {
            'layers': [16],
            'dropout_p': 0.2
        }
    }
    
    dataset = PdtiDataset(dataset_config)
    loader = DataLoader(dataset, batch_size=128, collate_fn=dataset.collate_fn)
    model = PdtiModel(model_config).to('cuda:0')
    
    times = {}
    for i, (dc, regrouper, p, m, aff) in enumerate(loader):
        a=time.time()
        dc = dc.to_device('cuda:0')
        p = p.to_device('cuda:0')
        m = m.to_device('cuda:0')
        b=time.time()
        if not 'to_device' in times:
            times['to_device'] = 0
        times['to_device'] += b-a
        if i == 10:
            break
        out = model(dc, regrouper, p, m, times=times) 
        
    print(times)
        
        
        