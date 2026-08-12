from pathlib import Path
import numpy as np
import torch
from model import SpikeCloneLSTM

DATA_PATH = Path("clone_data/binned_activity_masked.npy")
MODEL_PATH = Path("clone_data/clone_model_masked.pt")
SEED_WINDOW, SEGMENT, RUNS, BIN_SECONDS = 200, 200, 50, 0.05

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data = np.load(DATA_PATH)
model = SpikeCloneLSTM(num_electrodes=data.shape[1]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

traces = np.zeros((RUNS, SEGMENT), dtype="float32")
with torch.no_grad():
    for r in range(RUNS):
        s = np.random.randint(0, data.shape[0] - SEED_WINDOW)
        seed = torch.log1p(torch.from_numpy(data[s:s+SEED_WINDOW])).unsqueeze(0).to(device)
        log_rate_seq, hidden = model(seed)
        log_rate = log_rate_seq[:, -1, :]
        for i in range(SEGMENT):
            counts = torch.poisson(torch.exp(log_rate).clamp(min=0).cpu())
            traces[r, i] = counts.sum().item() / BIN_SECONDS
            nxt = torch.log1p(counts.to(device)).unsqueeze(1)
            log_rate_seq, hidden = model(nxt, hidden)
            log_rate = log_rate_seq[:, -1, :]

m = traces.mean(axis=0)
print(f"real:         {data.sum(axis=1).mean() / BIN_SECONDS:.2f}")
print(f"steps 1-10:   {m[:10].mean():.2f}")
print(f"steps 90-110: {m[90:110].mean():.2f}")
print(f"steps 190+:   {m[190:].mean():.2f}")
