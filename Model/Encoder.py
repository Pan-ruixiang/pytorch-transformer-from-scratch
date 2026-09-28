import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from FeedForward_Networks import PositionwiseFeedForward
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,src_mask):
        attn_output = self.self_attn(x,x,x,src_mask,self.dropout1)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x
