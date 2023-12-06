import torch.nn as nn
import torch.nn.functional as F

torch.set_default_device('cuda')

class LSTM(nn.Module):
    def __init__(self, vocab, dims, bidirection=False):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), dims)
        if bidirection:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.lstm = nn.LSTM(dims, dims, bidirectional=bidirection, batch_first=True)
        self.out = nn.Linear(dims * self.num_directions, len(vocab))
    
    def forward(self, src):
        embedded = self.embed(src)
        lstm_output, hidden = self.lstm(embedded)
        o = self.out(lstm_output)
        return o