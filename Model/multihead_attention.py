import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        super().__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
    def forward(self,q,k,v,mask = None,dropout = None):
        batch = q.size(0)
        Q = self.w_q(q).view(batch,-1,self.h,self.d_k).transpose(1,2)
        K = self.w_k(k).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch, -1, self.h, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0,-1e9)
        attn_scores = attn_scores.softmax(dim = -1)
        if dropout is not None:
            attn_scores = dropout(attn_scores)
        output = torch.matmul(attn_scores,V).transpose(1,2)
        x = output.contiguous().view(batch,-1,self.h * self.d_k)
        return self.w_o(x)
