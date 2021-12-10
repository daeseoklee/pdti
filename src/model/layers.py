from typing import Tuple, List, Dict, Any
from collections import OrderedDict as odict

import torch 
from torch import Tensor
import torch.nn as nn 

from asym.annotated_module import AnnotatedModule

from model.geometry import to_local

class FeedForward(AnnotatedModule):
    def __init__(self, prev_dim, dims, act_fn='relu', dropout_p=None):
        super().__init__()
        self.dims = [prev_dim] + dims
        self.last_dim = dims[-1]
        self.layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(len(self.dims) - 1)])
        activations = {
            'relu': nn.ReLU
        }
        self.activation = activations[act_fn]()
        self.use_dropout = (dropout_p is not None)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)
        self.weight_init_()
    def weight_init_(self):
        for layer in self.layers:
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_uniform_(layer.weight)
    def forward(self, x):
        assert x.shape[-1] == self.dims[0]
        for layer in self.layers:
            x = self.activation(layer(x))
            if self.use_dropout:
                x = self.dropout(x)
        return x 
    def get_mask_hint(self):
        return None
    def get_input_annot(self):
        return '(b, .., m_in)'
    def get_output_annot(self):
        return '(b, .., m_out)'
    
class LayerNorm(AnnotatedModule):
    def __init__(self, last_dim):
        super().__init__()
        self.last_dim = last_dim 
        self._layernorm = nn.LayerNorm(self.last_dim)
    def forward(self, x):
        return self._layernorm(x)
    def get_mask_hint(self):
        return None
    def get_input_annot(self):
        return '(b, .., m_last)'
    def get_output_annot(self):
        return '(b, .., m_last)'

class MaskedMaxpool(AnnotatedModule):
    def __init__(self, dim):
        """
        Args:
            dim ([type]): dimension to maxpool, expressed as a negative integer. It is assumed that every subsequent dimension is MDim(i.e. "model dimension")

        Raises:
            Exception: [description]
        """
        super().__init__()
        if dim >= 0:
            raise Exception('dim should be given as a negative integer')
        self.dim = dim
    def forward(self, x:Tensor, mask:Tensor):
        x = torch.where(mask, x, - torch.tensor(float('inf'), device=x.device))
        x = torch.max(x, dim=self.dim).values
        return x
    def get_mask_hint(self):
        return 'copy'
    def get_input_annot(self):
        #self.dim == -3, then (B, .., L_pool, M_1, M_2) 
        s = '(b, .., l_pool'
        for i in range(1, -self.dim):
            s += f', m_{i}'
        s += ')'
        return s
    def get_output_annot(self):
        s = '(b, ..'
        for i in range(1, -self.dim):
            s += f', m_{i}'
        s += ')'
        return s
    
class MultiModalRegressor(nn.Module):
    def __init__(self, last_dim, k):
        super().__init__()
        self.regressors = nn.ModuleList([nn.Linear(last_dim, 1) for _ in range(k)])
        self.weighters = nn.ModuleList([nn.Linear(last_dim, 1) for _ in range(k)])
        self.softmax = nn.Softmax(-1)
        self.weight_init_()
    def weight_init_(self):
        for regressor in self.regressors:
            nn.init.zeros_(regressor.bias)
            nn.init.xavier_uniform_(regressor.weight)
        for weighter in self.weighters:
            nn.init.zeros_(weighter.bias)
            nn.init.xavier_uniform_(weighter.weight)
    def forward(self, x):
        values = torch.stack([regressor(x) for regressor in self.regressors], dim=-1)
        weights = torch.stack([weighter(x) for weighter in self.weighters], dim=-1)
        weights = self.softmax(weights)
        return torch.sum(weights * values, dim=-1)
        
    
class MultiHeadRegressor(AnnotatedModule):
    def __init__(self, last_dim:int, metrics:List[str], prefix=None, ks=None, zero_weight=[]):
        super().__init__()
        self.last_dim = last_dim
        self.metrics = metrics
        self.prefix = prefix
        regressors = {}
        for metric in self.metrics:
            if ks is not None and metric in ks:
                k = ks[metric]
                regressors[metric] = MultiModalRegressor(self.last_dim, k)
            else:
                regressors[metric] = nn.Linear(self.last_dim, 1)
        self.regressors = nn.ModuleDict(regressors)
        self.zero_weight = zero_weight
        self.weight_init_()
    def weight_init_(self):
        for metric in self.metrics:
            if type(self.regressors[metric]) != nn.Linear:
                continue
            nn.init.zeros_(self.regressors[metric].bias)
            if metric in self.zero_weight:
                nn.init.zeros_(self.regressors[metric].weight)
            else:
                nn.init.xavier_uniform_(self.regressors[metric].weight) 
    def get_key_name(self, metric):
        if self.prefix is None:
            return metric
        return f'{self.prefix}_{metric}'
    
    def forward(self, x):
        return {self.get_key_name(metric): self.regressors[metric](x).squeeze(-1) for metric in self.metrics}
    def get_mask_hint(self):
        return None
    def get_input_annot(self):
        return '(b, .., m_hidden)'
    def get_output_annot(self):
        return {metric: '(b, ..)' for metric in self.metrics}
    
    
class RayAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=128, v_dim=12, num_heads=96, eps=1e-6, softplus=True, att_mode='linear'):
        super().__init__()
        self.eps = eps 
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.to_v = nn.Linear(hidden_dim, self.v_dim * self.num_heads)
        self.to_a = nn.Linear(hidden_dim, 3 * self.num_heads)
        self.to_b = nn.Linear(hidden_dim, 3 * self.num_heads)
        self.final_proj = nn.Linear((self.v_dim + 5) * self.num_heads, hidden_dim)
        if softplus:
            init_alphabeta = - 2 * torch.ones((1, 1, 1, num_heads))
        else:
            init_alphabeta = torch.zeros((1, 1, 1, num_heads))
        self.register_parameter('alpha', nn.Parameter(init_alphabeta))
        self.register_parameter('beta', nn.Parameter(init_alphabeta))
        if softplus:
            self.softplus = nn.Softplus()
        self.att_mode = att_mode
        self.att_softmax = nn.Softmax(-2) 
        
        self.weight_init_()
        
    def weight_init_(self):
        #zero bias
        nn.init.zeros_(self.to_v.bias)
        nn.init.constant_(self.to_a.bias, self.eps)
        nn.init.constant_(self.to_b.bias, self.eps)
        nn.init.zeros_(self.final_proj.bias)
        
        #queries, keys, values
        nn.init.xavier_uniform_(self.to_v.weight)
        
        #weights that lead to geometry
        nn.init.zeros_(self.to_a.weight)
        nn.init.zeros_(self.to_b.weight) 
        
        #layers right before residual additions 
        nn.init.zeros_(self.final_proj.weight)
        
    def get_alpha(self):
        if hasattr(self, 'softplus'):
            return self.softplus(self.alpha)
        return self.alpha
    
    def get_beta(self):
        if hasattr(self, 'softplus'):
            return self.softplus(self.beta)
        return self.beta
    
    def get_unnormalized_att_weights(self, r_size, theta):
        if self.att_mode == 'linear':
            return - self.get_alpha() * r_size - self.get_beta() * theta
        elif self.att_mode == 'theta-weighted':
            return - self.get_alpha() * r_size * theta - self.get_beta() * theta
        raise Exception('No such "att_mode" for RA')
    
        
    
    def forward(self, x:Tensor, R:Tensor, t:Tensor, mask:Tensor):
        #two residue dims: original and surrounding
        # v : surrounding vector
        assert len(x.shape) == 3 #(B, L_res, M_hidden)
        N = x.shape[0]
        L = x.shape[1]
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        assert mask.shape == (N, L)
        a = self.to_a(x).reshape(N, L, 1, self.num_heads, 3) #(B, L_res, 1, M_head, 3)
        b = self.to_b(x).reshape(N, L, 1, self.num_heads, 3) #(B, L_res, 1, M_head, 3)
        local = to_local(R, t, t) #(B, L_res, L_res, 3)
        r = local.unsqueeze(-2) - b #(B, L_res, L_res, M_head, 3)

        r_size = torch.sqrt(torch.sum(r ** 2, dim=-1)) #(B, L_res, L_res, M_head)
        a_size = torch.sqrt(torch.sum(a ** 2, dim=-1)) #(B, L_res, L_res, M_head)
        r_dot_a = torch.sum(r * a, dim=-1) #(B, L_res, L_res, M_head)
        theta = torch.acos(r_dot_a / ((r_size + self.eps ) * (a_size + self.eps))) #(B, L_res, L_res, M_head)

        #print(r_size.shape, self.alpha.shape)
        att_weights = self.get_unnormalized_att_weights(r_size, theta) #(B, L_res, L_res, M_head)
        off_diag = torch.logical_not(torch.eye(L, device=t.device, dtype=bool))
        att_mask = torch.logical_and(mask[:, None, :], off_diag[None, :, :]).unsqueeze(-1)
        neg_infty = - torch.tensor(float('inf'), device=t.device)
        att_weights = torch.where(att_mask, att_weights, neg_infty)
        att_weights = self.att_softmax(att_weights) #(B, L_res, L_res, M_head)
        v = self.to_v(x).reshape(N, 1, L, self.num_heads, self.v_dim) #(B, 1, L_res, M_head, M_v)
        
        v_aggr = torch.sum(att_weights.unsqueeze(-1) * v, dim=2) #(B, L_res, M_head, M_v)
        r_aggr = torch.sum(att_weights.unsqueeze(-1) * r, dim=2) #(B, L_res, M_head, 3)
        r_aggr_size = torch.sqrt(torch.sum(r_aggr ** 2, dim=-1)) #(B, L_res, M_head)
        theta_aggr = torch.sum(att_weights * theta, dim=2) #(B, L_res, M_head)
        
        concatenated = torch.cat([v_aggr.reshape(N, L, -1), r_aggr.reshape(N, L, -1), r_aggr_size, theta_aggr], dim=-1) 
        out = self.final_proj(concatenated) #(B, L_res, M_hidden)
        
        return out
        
        
        
class RayAttentionUnit(nn.Module):
    def __init__(self, hidden_dim=128, inter_dim=512, dropout_p=0.1, **kwargs):
        super().__init__() 
        self.hidden_dim = hidden_dim
        self.RA = RayAttentionLayer(hidden_dim=hidden_dim, **kwargs)
        self.layernorm_after_RA = nn.LayerNorm(hidden_dim)
        self.pointwise_ff = nn.Sequential(odict([
        ('linear1', nn.Linear(hidden_dim, inter_dim)),
        ('activation', nn.ReLU()),
        ('linear2', nn.Linear(inter_dim, hidden_dim))
        ]))
        self.layernorm_after_ff = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.weight_init_()
    
    def weight_init_(self):
        #zero bias
        nn.init.zeros_(self.pointwise_ff.linear1.bias)
        nn.init.zeros_(self.pointwise_ff.linear2.bias)
        
        #layers followed by relu: he initialization
        nn.init.kaiming_uniform_(self.pointwise_ff.linear1.weight)
        
        #layers right before residual connection
        nn.init.zeros_(self.pointwise_ff.linear2.weight)
        
    def forward(self, x, R, t, mask):
        assert x.shape[-1] == self.hidden_dim
        residual = x 
        x = self.dropout(self.RA(x, R, t, mask))
        x = self.layernorm_after_RA(residual + x)
        residual = x 
        x = self.dropout(self.pointwise_ff(x))
        x = self.layernorm_after_ff(residual + x)
        return x
        
        
    
class RayAttention(AnnotatedModule):
    def __init__(self, num_layers=1, **kwargs):
        super().__init__()
        self.units = nn.ModuleList([RayAttentionUnit(**kwargs) for _ in range(num_layers)])
    
    def forward(self, d, mask):
        x = d['x']
        R = d['R']
        t = d['t']
        for unit in self.units:
            x = unit(x, R, t, mask)
        return x 
    
    def get_mask_hint(self): #This will be changed soon
        return ['res']
    
    def get_input_annot(self):
        return {
            'x': '(b, l_res, m_hidden)',
            'R': '(b, l_res, 3, 3)',
            't': '(b, l_res, 3)'
        }
        
    def get_output_annot(self):
        return '(b, l_res, m_hidden)'
    


    