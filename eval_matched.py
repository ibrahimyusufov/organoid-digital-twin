from pathlib import Path
import sys
import numpy as np, torch
from corr_metric import mean_cross_corr
from evaluate import isi

DATA = Path("clone_data/binned_activity_masked.npy")
SEED_WINDOW, RESEED, STEPS, BIN_S, DRAWS = 200, 200, 60000, 0.05, 8

def rollout_at(model, data, start, steps, device):
    """Roll out, always reseeding from within [start, start+steps)."""
    out = np.zeros((steps, data.shape[1]), dtype="float32")
    t = 0
    with torch.no_grad():
        while t < steps:
            s = start + np.random.randint(0, max(1, steps - SEED_WINDOW))
            seed = torch.log1p(torch.from_numpy(data[s:s+SEED_WINDOW])).unsqueeze(0).to(device)
            lr_seq, hidden = model(seed)
            lr = lr_seq[:, -1, :]
            for _ in range(min(RESEED, steps - t)):
                c = torch.poisson(torch.exp(lr).clamp(min=0).cpu())
                out[t] = c[0].numpy(); t += 1
                lr_seq, hidden = model(torch.log1p(c.to(device)).unsqueeze(1), hidden)
                lr = lr_seq[:, -1, :]
    return out

def stats(x):
    r = x.sum(1) / BIN_S
    return dict(rate=r.mean(), fano=r.var()/r.mean(),
                silent=(x.sum(1)==0).mean(), corr=mean_cross_corr(x),
                isi_med=np.median(isi(x)))

if __name__ == "__main__":
    burst = "--burst" in sys.argv
    if burst:
        from model_burst import BurstCloneLSTM as Net
        path = Path("clone_data/clone_model_burst.pt")
    else:
        from model import SpikeCloneLSTM as Net
        path = Path("clone_data/clone_model_masked.pt")
    print(f"model: {path.name}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    real = np.load(DATA)
    model = Net(num_electrodes=real.shape[1]).to(device)
    model.load_state_dict(torch.load(path, map_location=device)); model.eval()

    rows = []
    for i in range(DRAWS):
        s = np.random.randint(0, len(real) - STEPS)
        rs, gs = stats(real[s:s+STEPS]), stats(rollout_at(model, real, s, STEPS, device))
        rows.append((rs, gs))
        print(f"draw {i+1}/{DRAWS}  hour {s*BIN_S/3600:5.1f}  "
              f"real corr {rs['corr']:.3f}  gen corr {gs['corr']:.3f}")

    print(f"\n{'metric':<10}{'real':>10}{'gen':>10}{'ratio':>10}")
    print("-"*40)
    for k in rows[0][0]:
        rv = np.mean([r[k] for r, _ in rows]); gv = np.mean([g[k] for _, g in rows])
        print(f"{k:<10}{rv:>10.3f}{gv:>10.3f}{gv/rv if rv else float('nan'):>10.2f}")
