import torch.nn as nn
from multihead_attention import MultiHeadAttention
from FeedForward_Networks import PositionwiseFeedForward
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.cross_attn = MultiHeadAttention(d_model,num_heads)
        self.Feed_Forward = PositionwiseFeedForward(d_model,d_ff,0.2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory
        attn_output = self.self_attn(x,x,x,tgt_mask,self.dropout)
        x = self.norm1(self.dropout(x + attn_output))
        attn_output = self.cross_attn(x,m,m,src_mask,self.dropout)
        x = self.norm2(self.dropout(x + attn_output))
        ffn_output = self.Feed_Forward(x)
        x = self.norm3(self.dropout(x + attn_output))
        return x

