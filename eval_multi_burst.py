from pathlib import Path
import numpy as np, torch
from model_burst import BurstCloneLSTM as SpikeCloneLSTM
from corr_metric import mean_cross_corr
from evaluate import rollout, isi

DATA = Path("clone_data/binned_activity_masked.npy")
MODEL = Path("clone_data/clone_model_burst.pt")
STEPS, BIN_S, DRAWS = 60000, 0.05, 8

def stats(x):
    r = x.sum(1) / BIN_S
    return dict(rate=r.mean(), fano=r.var()/r.mean(),
                silent=(x.sum(1)==0).mean(), corr=mean_cross_corr(x),
                isi_med=np.median(isi(x)))

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    real = np.load(DATA)
    model = SpikeCloneLSTM(num_electrodes=real.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device)); model.eval()

    R, G = [], []
    for i in range(DRAWS):
        s = np.random.randint(0, len(real) - STEPS)
        R.append(stats(real[s:s+STEPS]))
        G.append(stats(rollout(model, real, STEPS, device)))
        print(f"draw {i+1}/{DRAWS}")

    print(f"\n{'metric':<16}{'real mean':>12}{'real sd':>10}{'gen mean':>12}{'gen sd':>10}")
    print("-"*60)
    for k in R[0]:
        rv = np.array([d[k] for d in R]); gv = np.array([d[k] for d in G])
        print(f"{k:<16}{rv.mean():>12.3f}{rv.std():>10.3f}{gv.mean():>12.3f}{gv.std():>10.3f}")
