import torch.nn as nn

NUM_ELECTRODES = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2


class SpikeCloneLSTM(nn.Module):
    def __init__(self, num_electrodes=NUM_ELECTRODES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(num_electrodes, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_electrodes)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        log_rate = self.head(out)
        return log_rate, hidden
