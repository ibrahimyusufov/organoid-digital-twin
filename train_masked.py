from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from model import SpikeCloneLSTM

DATA_PATH = Path("clone_data/binned_activity_masked.npy")
MODEL_PATH = Path("clone_data/clone_model_masked.pt")
WINDOW, BATCH_SIZE, NUM_STEPS, LR = 200, 64, 3000, 1e-3
HOLDOUT_FRAC = 0.1

def sample_batch(data, window=WINDOW, batch_size=BATCH_SIZE):
    starts = np.random.randint(0, data.shape[0] - window - 1, size=batch_size)
    return torch.from_numpy(np.stack([data[s:s+window+1] for s in starts]))

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(DATA_PATH)
    split = int(len(data) * (1 - HOLDOUT_FRAC))
    train, val = data[:split], data[split:]
    print(f"device {device} | train {train.shape} | val {val.shape}")

    model = SpikeCloneLSTM(num_electrodes=data.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=True)

    for step in range(1, NUM_STEPS + 1):
        seq = sample_batch(train).to(device)
        log_rate, _ = model(torch.log1p(seq[:, :-1, :]))
        loss = loss_fn(log_rate, seq[:, 1:, :])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                v = sample_batch(val).to(device)
                vlr, _ = model(torch.log1p(v[:, :-1, :]))
                vloss = loss_fn(vlr, v[:, 1:, :]).item()
            model.train()
            print(f"step {step:>5}  train {loss.item():.4f}  val {vloss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")
