from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from model_latent import LatentBurstLSTM

DATA_PATH = Path("clone_data/binned_activity_masked.npy")
MODEL_PATH = Path("clone_data/clone_model_latent.pt")
WINDOW, BATCH_SIZE, NUM_STEPS, LR = 200, 64, 6000, 1e-3
HOLDOUT_FRAC, SAVE_EVERY = 0.1, 1000

def sample_batch(data, window=WINDOW, batch_size=BATCH_SIZE):
    starts = np.random.randint(0, data.shape[0] - window - 1, size=batch_size)
    return torch.from_numpy(np.stack([data[s:s+window+1] for s in starts]))

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(DATA_PATH)
    split = int(len(data) * (1 - HOLDOUT_FRAC))
    train, val = data[:split], data[split:]
    print(f"device {device} | train {train.shape} | val {val.shape}")

    model = LatentBurstLSTM(num_electrodes=data.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=True)

    for step in range(1, NUM_STEPS + 1):
        seq = sample_batch(train).to(device)
        log_rate, _ = model(torch.log1p(seq[:, :-1, :]), sample_latent=True)
        loss = loss_fn(log_rate, seq[:, 1:, :])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                v = sample_batch(val).to(device)
                vlr, _ = model(torch.log1p(v[:, :-1, :]), sample_latent=True)
                vloss = loss_fn(vlr, v[:, 1:, :]).item()
            model.train()
            sd = torch.nn.functional.softplus(model.drive_sd.bias).item()
            print(f"step {step:>5}  train {loss.item():.4f}  val {vloss:.4f}  sd_bias {sd:.3f}")

        if step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), MODEL_PATH)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"gain: {model.gain.detach().cpu().numpy().round(2)}")
    print(f"saved {MODEL_PATH}")
