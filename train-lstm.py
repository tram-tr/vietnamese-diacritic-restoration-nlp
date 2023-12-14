import torch.nn as nn
import torch.nn.functional as F
import pickle
import utils
import argparse
import dataloader
import torch.utils

# torch.set_default_device('cuda')

train_tone = 'data/train.tone'
train_notone = 'data/train.notone'
dev_tone = 'data/dev.tone'
dev_notone = 'data/dev.notone'
test_tone = 'data/test.tone'
test_notone = 'data/test.notone'

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

class LSTM(nn.Module):
    def __init__(self, src_vocab, trg_vocab, dims):
        super().__init__()
        self.embed = nn.Embedding(len(src_vocab), dims)

        self.lstm = nn.LSTM(dims, dims, batch_first=True)
        self.out = nn.Linear(dims * self.num_directions, len(trg_vocab))
    
    def forward(self, src):
        embedded = self.embed(src)
        o, h = self.lstm(embedded)
        o = self.out(o)
        return o
    
if __name__=='__main__':
    train_dataset = utils.create_dataset(train_notone, train_tone)
    train_iter = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True)

    dev_dataset = utils.create_dataset(dev_notone, dev_tone)
    dev_iter = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=32, shuffle=True)