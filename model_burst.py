import torch
import torch.nn as nn

class BurstCloneLSTM(nn.Module):
    """LSTM emitting per-channel log-rates plus a shared network drive."""
    def __init__(self, num_electrodes=13, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_electrodes, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_electrodes)
        self.drive = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        log_rate = self.head(out) + self.drive(out)
        return log_rate, hidden
