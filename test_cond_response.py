from pathlib import Path
import numpy as np, torch
from model_cond import CondCloneLSTM

SPIKES = Path("clone_data/binned_activity_masked.npy")
MODEL = Path("clone_data/clone_model_cond.pt")
LIVE = np.load("clone_data/live_channels.npy")
SEED_W, POST, TRIALS, AMP = 200, 10, 300, 10.0

PATTERNS = {"B": (10, 11, 13, 15), "C": (1, 9, 12, 14)}

def run(model, data, stim_ch, device):
    idx = [list(LIVE).index(c) for c in stim_ch if c in LIVE]
    acc = np.zeros(len(LIVE))
    with torch.no_grad():
        for _ in range(TRIALS):
            s = np.random.randint(0, len(data) - SEED_W)
            sp = torch.log1p(torch.from_numpy(data[s:s+SEED_W])).unsqueeze(0).to(device)
            zero = torch.zeros_like(sp)
            lr, hidden = model(sp, zero, sample_latent=True)
            lr = lr[:, -1, :]
            for t in range(POST):
                st = torch.zeros(1, 1, len(LIVE), device=device)
                if t == 0:
                    for i in idx:
                        st[0, 0, i] = np.log1p(AMP)
                c = torch.poisson(torch.exp(lr).clamp(min=0).cpu())
                acc += c[0].numpy()
                lr, hidden = model(torch.log1p(c.to(device)).unsqueeze(1), st,
                                   hidden, sample_latent=True)
                lr = lr[:, -1, :]
    return acc / TRIALS

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(SPIKES)
    model = CondCloneLSTM(num_electrodes=data.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device)); model.eval()

    res = {k: run(model, data, v, device) for k, v in PATTERNS.items()}
    print("clone response, spikes per trial over 0.5 s after stim:")
    print(f"{'ch':<6}{'B':>10}{'C':>10}   (real: B->ch15, C->ch8/9)")
    for i, ch in enumerate(LIVE):
        print(f"{ch:<6}{res['B'][i]:>10.3f}{res['C'][i]:>10.3f}")
    v = np.array([res['B'], res['C']])
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    print(f"\ncosine similarity B vs C: {vn[0] @ vn[1]:.3f}   (real: 0.458)")
