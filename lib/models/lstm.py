import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, init_w=0.003, hidden=300, embedding_size=100):

        self.init_w = init_w
        self.hidden = hidden
        self.embedding_size = embedding_size

        self.ln_1 = nn.LayerNorm(self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden, batch_first=True)
        self.ln_2 = nn.LayerNorm(self.hidden)
        self.linear = nn.Linear(self.hidden, nb_actions)

    def forward(self, x):
        x = self.ln_1(x)
        x = self.lstm(x)
        x = self.ln_2(x)
        return self.linear(x)


    
        
