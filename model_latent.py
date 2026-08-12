import torch
import torch.nn as nn

class LatentBurstLSTM(nn.Module):
    """Per-channel log-rates plus a shared latent drive with learned per-channel gain.

    At generation time a single noise sample per bin is drawn and scaled by
    per-channel weights, so channels rise and fall together rather than
    independently.
    """
    def __init__(self, num_electrodes=13, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_electrodes, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_electrodes)
        self.drive_mu = nn.Linear(hidden_size, 1)
        self.drive_sd = nn.Linear(hidden_size, 1)
        self.gain = nn.Parameter(torch.ones(num_electrodes))

    def forward(self, x, hidden=None, sample_latent=False):
        out, hidden = self.lstm(x, hidden)
        base = self.head(out)
        mu = self.drive_mu(out)
        if sample_latent:
            sd = torch.nn.functional.softplus(self.drive_sd(out)) + 1e-3
            z = mu + sd * torch.randn_like(mu)
        else:
            z = mu
        return base + z * self.gain, hidden
