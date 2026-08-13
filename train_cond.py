from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from model_cond import CondCloneLSTM

SPIKES = Path("clone_data/binned_activity_masked.npy")
STIM = Path("clone_data/stim_matrix.npy")
MODEL_PATH = Path("clone_data/clone_model_cond.pt")
WINDOW, BATCH, STEPS, LR = 200, 64, 6000, 1e-3
HOLDOUT, SAVE_EVERY = 0.1, 1000

def sample_batch(sp, sm, window=WINDOW, batch=BATCH):
    s = np.random.randint(0, sp.shape[0] - window - 1, size=batch)
    a = np.stack([sp[i:i+window+1] for i in s])
    b = np.stack([sm[i:i+window+1] for i in s])
    return torch.from_numpy(a), torch.from_numpy(b)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sp, sm = np.load(SPIKES), np.load(STIM)
    assert sp.shape == sm.shape, (sp.shape, sm.shape)
    cut = int(len(sp) * (1 - HOLDOUT))
    tr_sp, tr_sm, va_sp, va_sm = sp[:cut], sm[:cut], sp[cut:], sm[cut:]
    print(f"device {device} | train {tr_sp.shape} | val {va_sp.shape}")

    model = CondCloneLSTM(num_electrodes=sp.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=True)

    for step in range(1, STEPS + 1):
        a, b = sample_batch(tr_sp, tr_sm)
        a, b = a.to(device), b.to(device)
        lr_, _ = model(torch.log1p(a[:, :-1]), torch.log1p(b[:, :-1]), sample_latent=True)
        loss = loss_fn(lr_, a[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                va, vb = sample_batch(va_sp, va_sm)
                va, vb = va.to(device), vb.to(device)
                vl, _ = model(torch.log1p(va[:, :-1]), torch.log1p(vb[:, :-1]), sample_latent=True)
                vloss = loss_fn(vl, va[:, 1:]).item()
            model.train()
            print(f"step {step:>5}  train {loss.item():.4f}  val {vloss:.4f}")

        if step % SAVE_EVERY == 0:
            torch.save(model.state_dict(), MODEL_PATH)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")
