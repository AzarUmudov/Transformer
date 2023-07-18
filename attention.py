import torch
import torch.nn as nn
import math
import embedding as em

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model%h == 0, "d_model should be divisible by h"
        self.h = h
        self.d_k = d_model//h
        self.wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.wv = nn.Linear(in_features=d_model, out_features=d_model)
        self.wo = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, q, k, v, mask=None):
        query = self.wq(q) 
        key = self.wk(k)
        value = self.wv(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        attention = (query @ key.transpose(-2, -1))/math.sqrt(self.d_model)

        if mask is not None:
            attention.masked_fill_(mask==0, '-1e9')
            
        attention = torch.softmax(attention, dim=-1) @ value 
        attention = attention.transpose(1,2).reshape(attention.shape[0], -1, self.d_k*self.h)
        x = self.wo(attention)
        return x
    

atn = MultiHeadAttention(512, 8)
random_tokens = torch.randint(low=0, high=5000, size=(6,)).unsqueeze(0)
inp_emb = em.InputEmbedding(5000, 512)(random_tokens)
pos_emb = em.PositionEncoding(d_model=512, seq_len=6, dropout=0.1)(inp_emb)
x = atn(pos_emb, pos_emb, pos_emb)
print(f'Attention output: {x.shape}')