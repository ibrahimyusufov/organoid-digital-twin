import torch
import torch.nn as nn

class CondCloneLSTM(nn.Module):
    """Spike history + stimulation input, with shared latent drive."""
    def __init__(self, num_electrodes=13, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(num_electrodes * 2, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_electrodes)
        self.drive_mu = nn.Linear(hidden_size, 1)
        self.drive_sd = nn.Linear(hidden_size, 1)
        self.gain = nn.Parameter(torch.ones(num_electrodes))

    def forward(self, spikes, stim, hidden=None, sample_latent=False):
        x = torch.cat([spikes, stim], dim=-1)
        out, hidden = self.lstm(x, hidden)
        mu = self.drive_mu(out)
        if sample_latent:
            sd = torch.nn.functional.softplus(self.drive_sd(out)) + 1e-3
            z = mu + sd * torch.randn_like(mu)
        else:
            z = mu
        return self.head(out) + z * self.gain, hidden
