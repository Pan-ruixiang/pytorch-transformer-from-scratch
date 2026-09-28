import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from Model_Architecture import Transformer
src_vocab = 2867
tgt_vocab = 1832
d_model = 256
num_heads = 4
num_layers = 3
d_ff = 1024
max_seq_length = 64
dropout = 0.2
batch_size = 32
from Tokenizer import Vocab,read_parallel_data
en_data,zh_data = read_parallel_data("cmn.txt")
en_vocab = Vocab(en_data)
#print(len(en_vocab.word2idx))
zh_vocab = Vocab(zh_data)
#print(len(zh_vocab.idx2word))
class TranslationDataset(Dataset):
    def __init__(self,en_data,zh_data,en_vocab,zh_vocab,max_len = 64):
        self.en_data = en_data
        self.zh_data = zh_data
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.en_data)
    def __getitem__(self,idx):
        en_sent = self.en_data[idx]
        zh_sent = self.zh_data[idx]
        en_ids = self.en_vocab.encode(en_sent)
        zh_ids = self.zh_vocab.encode(zh_sent)
        en_ids = en_ids[:self.max_len] + [0] * (self.max_len - len(en_ids))
        zh_ids = zh_ids[:self.max_len] + [0] * (self.max_len - len(zh_ids))
        return torch.tensor(en_ids,dtype = torch.long),torch.tensor(zh_ids,dtype = torch.long)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = TranslationDataset(en_data,zh_data,en_vocab,zh_vocab,64)
dataloader = DataLoader(datasets,batch_size = 2,shuffle = True)
model = Transformer(src_vocab,tgt_vocab,d_model,num_heads,num_layers,d_ff,max_seq_length,dropout)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss(ignore_index = 0)
optimizer = torch.optim.Adam(model.parameters(),betas = (0.9,0.98),eps = 1e-9,lr = 0.001)
model.train()
for epoch in range(80):
    total_loss = 0
    for src,tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        enc_output = model.encode(src)
        tgt_mask = model.generate_src_mask(src).to(device)
        dec_output = model.decode(enc_output,tgt[:,:-1],tgt_mask)
        optimizer.zero_grad()
        output = model(dec_output)
        loss = criterion(output.reshape(-1,tgt_vocab),tgt[:,1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch:{epoch+1},Loss:{total_loss / len(dataloader)}")



