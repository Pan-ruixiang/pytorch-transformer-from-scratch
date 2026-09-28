import torch
class Vocab:
    def __init__(self,sentences,min_freq = 1):
        self.word2idx = {"<PAD>":0,"<SOS>":1,"<EOS>":2,"UNK":3}
        self.idx2word = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"UNK"}
        self.build_vocab(sentences,min_freq)
    def build_vocab(self,sentences,min_freq):
        freq = {}
        for sent in sentences:
            for word in sent:
                freq[word] = freq.get(word,0) + 1
        idx = 4
        for word,cnt in freq.items():
            if cnt >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    def encode(self,sentence):
        return [self.word2idx.get(word,3) for word in ["<SOS>"] + sentence + ["<EOS>"]]
    def decode(self,idx_list):
        return [self.idx2word.get(idx,"<UNK>") for idx in idx_list]

def read_parallel_data(path):
    en_sents = []
    zh_sents = []
    with open(path,"r",encoding = "UTF-8") as f:
        for i,line in enumerate(f):
            if i >= 5000:
                break
            line = line.strip()
            if not line:
                continue
            en,zh,_ = line.split("\t")
            en_tokens = en.lower().split()
            zh_tokens = list(zh)
            en_sents.append(en_tokens)
            zh_sents.append(zh_tokens)
    return en_sents,zh_sents
en_data,zh_data = read_parallel_data("cmn.txt")
en_vocab = Vocab(en_data)
print(len(en_vocab.word2idx))
zh_vocab = Vocab(zh_data)
print(len(zh_vocab.idx2word))