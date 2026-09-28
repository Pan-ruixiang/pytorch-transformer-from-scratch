import torch,copy
import torch.nn as nn
from Embeddings import Embeddings
from Positional_Encoding import PositionalEncoding
from Encoder import EncoderLayer
from Decoder import DecoderLayer
class Transformer(nn.Module):
    def __init__(self,src_vocab,tgt_vocab,d_model,num_heads,num_layers,d_ff,
                 max_seq_length,dropout):
        super().__init__()
        self.encoder_embedding = Embeddings(d_model,src_vocab)
        self.decoder_embedding = Embeddings(d_model,tgt_vocab)
        self.positional_encoding = PositionalEncoding(d_model,dropout,max_seq_length)

        self.encoder = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model,num_heads,d_ff,dropout)) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([copy.deepcopy(DecoderLayer(d_model,num_heads,d_ff,dropout)) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model,tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def generate_src_mask(self,src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to("cuda")
        return src_mask

    def generate_tgt_mask(self,tgt):
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2).to("cuda")
        seq_length = tgt.size(1)
        look_ahead_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to("cuda")
        tgt_mask = (tgt_mask & look_ahead_mask)
        return tgt_mask


    def encode(self,src):
        src_mask = self.generate_src_mask(src).to("cuda")
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for encoder_layer in self.encoder:
            enc_output = encoder_layer(enc_output,src_mask)
        return enc_output

    def decode(self,memory,tgt,src_mask):
        tgt_mask = self.generate_tgt_mask(tgt).to("cuda")
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for decoder_layer in self.decoder:
            dec_output = decoder_layer(dec_output, memory, src_mask, tgt_mask)
        return dec_output

    def forward(self,x):
        return self.fc(x)





