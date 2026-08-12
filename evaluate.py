from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import SpikeCloneLSTM
from corr_metric import mean_cross_corr

DATA = Path("clone_data/binned_activity_masked.npy")
MODEL = Path("clone_data/clone_model_masked.pt")
FIG = Path("figures/evaluation_masked.png")
SEED_WINDOW, RESEED, STEPS, BIN_S = 200, 200, 60000, 0.05

def rollout(model, data, steps, device):
    out = np.zeros((steps, data.shape[1]), dtype="float32")
    t = 0
    with torch.no_grad():
        while t < steps:
            s = np.random.randint(0, data.shape[0] - SEED_WINDOW)
            seed = torch.log1p(torch.from_numpy(data[s:s+SEED_WINDOW])).unsqueeze(0).to(device)
            lr_seq, hidden = model(seed)
            lr = lr_seq[:, -1, :]
            for _ in range(min(RESEED, steps - t)):
                c = torch.poisson(torch.exp(lr).clamp(min=0).cpu())
                out[t] = c[0].numpy(); t += 1
                lr_seq, hidden = model(torch.log1p(c.to(device)).unsqueeze(1), hidden)
                lr = lr_seq[:, -1, :]
    return out

def isi(x):
    """Inter-spike intervals in bins, pooled across channels."""
    gaps = []
    for ch in range(x.shape[1]):
        idx = np.flatnonzero(x[:, ch] > 0)
        if len(idx) > 1:
            gaps.append(np.diff(idx))
    return np.concatenate(gaps) if gaps else np.array([1])

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    real = np.load(DATA)
    model = SpikeCloneLSTM(num_electrodes=real.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    gen = rollout(model, real, STEPS, device)
    _st = np.random.randint(0, len(real) - STEPS); real_s = real[_st:_st+STEPS]

    print(f"{'metric':<28}{'real':>10}{'generated':>12}")
    print("-" * 50)
    rr, gr = real_s.sum(1)/BIN_S, gen.sum(1)/BIN_S
    print(f"{'mean rate (Hz)':<28}{rr.mean():>10.2f}{gr.mean():>12.2f}")
    print(f"{'std of rate':<28}{rr.std():>10.2f}{gr.std():>12.2f}")
    print(f"{'fano factor':<28}{rr.var()/rr.mean():>10.2f}{gr.var()/gr.mean():>12.2f}")
    print(f"{'silent bin fraction':<28}{(real_s.sum(1)==0).mean():>10.3f}{(gen.sum(1)==0).mean():>12.3f}")
    ri, gi = isi(real_s), isi(gen)
    print(f"{'median ISI (bins)':<28}{np.median(ri):>10.1f}{np.median(gi):>12.1f}")
    print(f"{'p99 ISI (bins)':<28}{np.percentile(ri,99):>10.1f}{np.percentile(gi,99):>12.1f}")

    print(f"{'mean cross-ch corr':<28}{mean_cross_corr(real_s):>10.3f}{mean_cross_corr(gen):>12.3f}")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0,0].plot(rr[:2000], lw=.7, label="real"); ax[0,0].plot(gr[:2000], lw=.7, alpha=.7, label="gen")
    ax[0,0].set_title("Population rate (100 s)"); ax[0,0].legend()
    bins = np.arange(0, 60)
    ax[0,1].hist(ri, bins=bins, density=True, alpha=.6, label="real")
    ax[0,1].hist(gi, bins=bins, density=True, alpha=.6, label="gen")
    ax[0,1].set_yscale("log"); ax[0,1].set_title("ISI distribution"); ax[0,1].legend()
    w = np.arange(real.shape[1])
    ax[1,0].bar(w-.2, real_s.mean(0)/BIN_S, .4, label="real")
    ax[1,0].bar(w+.2, gen.mean(0)/BIN_S, .4, label="gen")
    ax[1,0].set_title("Per-channel rate"); ax[1,0].legend()
    _live = (real_s.std(0) > 0) & (gen.std(0) > 0)
    _rc = np.corrcoef(real_s[:, _live].T); _gc = np.corrcoef(gen[:, _live].T)
    _iu = np.triu_indices(_live.sum(), 1)
    ax[1,1].scatter(_rc[_iu], _gc[_iu], s=12)
    lim = [min(_rc[_iu].min(), _gc[_iu].min()), max(_rc[_iu].max(), _gc[_iu].max())]
    ax[1,1].plot(lim, lim, "k--", lw=.8); ax[1,1].set_title("Cross-channel corr: real vs gen")
    ax[1,1].set_xlabel("real"); ax[1,1].set_ylabel("generated")
    fig.tight_layout(); FIG.parent.mkdir(exist_ok=True); fig.savefig(FIG, dpi=150)
    print(f"\nsaved {FIG}")
