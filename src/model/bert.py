import torch 
import torch.nn as nn 
from vocab import PR_VOCAB, SM_VOCAB
from collections import OrderedDict
from typing import Tuple, List, Dict
import json

__author__ = 'Daeseok Lee'

GELU_GAIN = 1.85

def print_stats(name, x):
    mean = torch.mean(x).item()
    std = torch.std(x).item()
    print(f'{name} - ({mean - std}~{mean + std})')

class BertLayer(nn.Module):
    def __init__(self, num_layers, dim_hidden, dim_intermediate, num_att_heads, dim_key, dim_val, biased_qkv=False, skip_weight_init=False, do_print_stats=False):
        super().__init__()
        self.num_layers = num_layers #required for initialization 
        self.dim_hidden = dim_hidden
        self.dim_intermediate = dim_intermediate
        self.num_att_heads = num_att_heads
        self.dim_key = dim_key
        self.dim_val = dim_val
        self.dim_query = dim_key
        self.biased_qkv = biased_qkv
        self.to_query = nn.Linear(self.dim_hidden, self.num_att_heads * self.dim_query, bias = self.biased_qkv) 
        self.to_key = nn.Linear(self.dim_hidden, self.num_att_heads * self.dim_key, bias = self.biased_qkv) 
        self.to_val = nn.Linear(self.dim_hidden, self.num_att_heads * self.dim_val, bias = self.biased_qkv)
        self.softmax = nn.Softmax(dim = -1)
        self.proj = nn.Linear(self.num_att_heads * self.dim_val, self.dim_hidden, bias = self.biased_qkv)
        self.dropout_after_att = nn.Dropout(p = 0.1)
        self.layernorm_after_att = nn.LayerNorm(self.dim_hidden)
        self.pointwise_ff = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(self.dim_hidden, self.dim_intermediate)),
        ('activation', nn.GELU()),
        ('linear2', nn.Linear(self.dim_intermediate, self.dim_hidden))
        ]))

        self.dropout_after_ff = nn.Dropout(p = 0.1)
        self.layernorm_after_ff = nn.LayerNorm(self.dim_hidden)

        if not skip_weight_init:
            self.weight_init_()

        self.do_print_stats = do_print_stats

    def weight_init_(self):
        if self.biased_qkv:
            nn.init.zeros_(self.to_query.bias)
            nn.init.zeros_(self.to_key.bias)
            nn.init.zeros_(self.to_val.bias)
            nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.pointwise_ff.linear1.bias)
        nn.init.zeros_(self.pointwise_ff.linear2.bias)
        nn.init.xavier_normal_(self.to_query.weight)
        nn.init.xavier_normal_(self.to_key.weight)
        nn.init.xavier_normal_(self.to_val.weight)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.xavier_normal_(self.pointwise_ff.linear1.weight, gain=GELU_GAIN)
        nn.init.xavier_normal_(self.pointwise_ff.linear2.weight)

    def forward(self, x : torch.Tensor, attention_mask = None):
        residual = x
        #(batch, seq, hidden)
        query = self.to_query(x).reshape(x.shape[0], x.shape[1], self.num_att_heads, self.dim_query).transpose(1, 2) #(batch, head, next_seq, query)
        key = self.to_key(x).reshape(x.shape[0], x.shape[1], self.num_att_heads, self.dim_key).transpose(1, 2) #(batch, head, prev_seq, key)
        score = torch.matmul(query, key.transpose(2,3)) * (self.dim_key) ** (-0.5) #(batch, head, next_seq, prev_seq)
        if attention_mask is not None:
            score -= 100.0 * torch.logical_not(attention_mask[:, None, None, :])
        if self.do_print_stats:
            print_stats('query', query)
            print_stats('key', key)
            print_stats('score_logit', score)
        score = self.softmax(score) #(batch, head, next_seq, prev_seq)
        if self.do_print_stats:
            print_stats('score_log10_prob', torch.log10(score))
            print_stats('score_min_log10_prob', torch.min(torch.log10(score), dim=-1).values)
            print_stats('score_max_log10_prob', torch.max(torch.log10(score), dim=-1).values)
        val = self.to_val(x).reshape(x.shape[0], x.shape[1], self.num_att_heads, self.dim_val).transpose(1, 2) #(batch, head, prev_seq, val)
        x = torch.matmul(score, val).transpose(1, 2) #(batch, seq, head, val)
        if self.do_print_stats:
            print_stats('val', val)
            print_stats('weighted_sum', x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]) #(batch, seq, head * val)
        x = self.proj(x) #(batch, seq, hidden)
        x = self.dropout_after_att(x)
        x = x + residual
        x = self.layernorm_after_att(x)
        residual = x
        if self.do_print_stats:
            print_stats('before_ff', x)
        x = self.pointwise_ff(x) #(batch, seq, hidden)
        if self.do_print_stats:
            print_stats('ff', x)
        x = self.dropout_after_ff(x)
        x = x + residual 
        x = self.layernorm_after_ff(x)
        return x
   
    

class Bert(nn.Module):
    def __init__(self, max_seq_len=256, size_vocab=len(PR_VOCAB), dim_hidden=768, dim_intermediate=1024, num_att_heads=12, num_layers=12, dim_query=64, biased_qkv=False, skip_weight_init=False, do_print_stats=False):
        super().__init__()
        self.num_pos = max_seq_len + 3
        self.size_vocab = size_vocab 
        self.dim_hidden = dim_hidden
        self.dim_intermediate = dim_intermediate
        self.num_att_heads = num_att_heads
        self.dim_key = dim_query
        self.dim_val = dim_hidden // num_att_heads
        self.dim_query = dim_query
        self.pos_emb = nn.Embedding(self.num_pos, dim_hidden) #initialized from N(0,1)
        self.vocab_emb = nn.Embedding(size_vocab, dim_hidden) #initialized from N(0,1)
        self.dropout_after_emb = nn.Dropout(p = 0.1)
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
        [BertLayer(num_layers, self.dim_hidden, self.dim_intermediate, self.num_att_heads, self.dim_key, self.dim_val, biased_qkv=biased_qkv, skip_weight_init=skip_weight_init, do_print_stats=do_print_stats) for _ in range(num_layers)]
        )
        if not skip_weight_init:
            self.weight_init_()
        self.do_print_stats = do_print_stats
    def weight_init_(self):
        self.pos_emb.weight.data.multiply_(2 ** -0.5)
        self.vocab_emb.weight.data.multiply_(2 ** -0.5)
    def forward(self, x, attention_mask = None):
        #(batch, seq)
        assert len(x.shape) == 2
        x = self.vocab_emb(x) + self.pos_emb.weight[None, :x.shape[1], :] #(batch, seq, hidden)
        if self.do_print_stats:
            print_stats('embedding', x)
        x = self.dropout_after_emb(x)
        for layer in self.layers:
            x = layer(x, attention_mask = attention_mask) #(batch, seq, hidden)
        return x
    def load_pretrained(self, path, map_location=None):
        print(f'loading pretrained bert...')
        d = torch.load(path, map_location=map_location)['state_dict']
        new_d = {}
        for name, weight in d.items():
            if name.startswith('bert_mlm.bert.'):
                new_d[name[14:]] = weight

        self.load_state_dict(new_d)
        print('loaded pretraind bert!!!')


    



class BertMLM(nn.Module):
    def __init__(self, bert : Bert):
        super().__init__()
        self.bert = bert
        self.dim_hidden = bert.dim_hidden
        self.size_vocab = bert.size_vocab
        self.final_classifier = nn.Sequential(OrderedDict([
        ('linear1', nn.Linear(self.dim_hidden, self.dim_hidden)),
        ('activation', nn.GELU()),
        ('layernorm', nn.LayerNorm(self.dim_hidden)),
        ('linear2', nn.Linear(self.dim_hidden, self.size_vocab))
        ]))


    def weight_init_(self):
        nn.init.zeros_(self.final_classifier.linear1.bias)
        nn.init.zeros_(self.final_classifier.linear2.bias)
        nn.init.xavier_normal_(self.final_classifier.linear1.weight, gain=GELU_GAIN)
        nn.init.xavier_normal_(self.final_classifier.linear2.weight)

    def forward(self, x, attention_mask = None):
        return self.final_classifier(self.bert(x, attention_mask = attention_mask))
    
    def load_pretrained(self, path, map_location=None):
        print(f'loading pretrained bert mlm...')
        d = torch.load(path, map_location=map_location)['state_dict']
        new_d = {}
        for name, weight in d.items():
            if name.startswith('bert_mlm.'):
                new_d[name[9:]] = weight
        self.load_state_dict(new_d)
        print('finished loading.')

def load_pretrained_bert(bert_config, path, map_location=None):
    print(f'loading pretrained bert...')
    bert = Bert(**bert_config, skip_weight_init=True)
    d = torch.load(path, map_location=map_location)['state_dict']
    new_d = {}
    for name, weight in d.items():
        if name.startswith('bert_mlm.bert.'):
            new_d[name[14:]] = weight

    bert.load_state_dict(new_d)
    print('finished loading')

    return bert

def load_pretrained_bert_mlm(bert_config, path, map_location=None):
    print(f'loading pretrained bert mlm...')
    bert = Bert(**bert_config, skip_weight_init=True)
    bert_mlm = BertMLM(bert)
    d = torch.load(path, map_location=map_location)['state_dict']
    new_d = {}
    for name, weight in d.items():
        if name.startswith('bert_mlm.'):
            new_d[name[9:]] = weight
    bert_mlm.load_state_dict(new_d)
    
    return bert_mlm

def test_stats():
    length = 200
    input_ids = torch.randint(0, len(PR_VOCAB), (1, length))
    bert = Bert(num_pos = 258, size_vocab = len(PR_VOCAB), dim_hidden = 1024, dim_intermediate = 4096, num_att_heads = 16, num_layers = 30, dim_key = 64, dim_val = 64, biased_qkv = True, skip_weight_init = False, do_print_stats=True)
    print('created model!')
    bert_mlm = BertMLM(bert)
    bert_mlm.eval()
    
    attention_mask = torch.ones((1, length), dtype=torch.int)

    softmax = torch.nn.Softmax(-1)
    o = softmax(bert_mlm(input_ids, attention_mask = attention_mask))
    print(o.shape)



if __name__ == '__main__':
    test_stats()