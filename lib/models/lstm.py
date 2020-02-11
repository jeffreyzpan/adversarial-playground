import torch
import torch.nn as nn

class StockLSTM(nn.Module):

    def __init__(self, input_size=5, hidden=300):

        super(StockLSTM, self).__init__()
        self.hidden = hidden
        self.input_size = input_size

        self.ln_1 = nn.LayerNorm(self.input_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden, batch_first=True)
        self.ln_2 = nn.LayerNorm(self.hidden)
        self.linear = nn.Linear(self.hidden, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ln_1(x)
        x, hidden = self.lstm(x)
        x = self.ln_2(x)
        x = self.linear(x)
        return self.relu(x)


def stock_lstm(input_size=5, hidden=300):
    model = StockLSTM(input_size, hidden)
    return model
        
