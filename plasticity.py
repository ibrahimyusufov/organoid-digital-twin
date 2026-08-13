from pathlib import Path
import numpy as np
import torch
from model_cond import CondCloneLSTM

SPIKES = Path("clone_data/binned_activity_masked.npy")
MODEL = Path("clone_data/clone_model_cond.pt")
LIVE = np.load("clone_data/live_channels.npy")

PATTERNS = {"B": (10, 11, 13, 15), "C": (1, 9, 12, 14)}
SEED_W, POST, AMP = 200, 10, 10.0
TAU, LR_PLAST = 0.7, 0.05


class PlasticReadout:
    def __init__(self, n_ch, labels):
        self.labels = list(labels)
        self.w = np.ones((len(labels), n_ch), dtype="float32") * 0.1
        self.trace = np.zeros_like(self.w)

    def decide(self, resp):
        return self.labels[int(np.argmax(self.w @ resp))]

    def accumulate(self, li, gi, resp):
        self.trace *= TAU
        r = resp / (np.linalg.norm(resp) + 1e-9)
        self.trace[li] += r
        if gi != li:
            self.trace[gi] -= r

    def apply_reward(self, r):
        self.w += LR_PLAST * r * self.trace



def evoke(model, data, stim_ch, device):
    idx = [list(LIVE).index(c) for c in stim_ch if c in LIVE]
    acc = np.zeros(len(LIVE), dtype="float32")
    with torch.no_grad():
        s = np.random.randint(0, len(data) - SEED_W)
        sp = torch.log1p(torch.from_numpy(data[s:s+SEED_W])).unsqueeze(0).to(device)
        lr, hidden = model(sp, torch.zeros_like(sp), sample_latent=True)
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
    return acc


def session(model, data, device, trials=600, shuffled=False, seed=0):
    rng = np.random.default_rng(seed)
    ro = PlasticReadout(len(LIVE), PATTERNS)
    correct = []
    for t in range(trials):
        true = ro.labels[rng.integers(len(ro.labels))]
        resp = evoke(model, data, PATTERNS[true], device)
        guess = ro.decide(resp)
        ok = guess == true
        correct.append(ok)
        ro.accumulate(ro.labels.index(true), ro.labels.index(guess), resp)
        r = rng.choice([1.0, -1.0]) if shuffled else (0.3 if ok else 1.0)
        ro.apply_reward(r)
    return np.array(correct)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(SPIKES)
    model = CondCloneLSTM(num_electrodes=data.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    for tag, shuf in [("real reward", False), ("shuffled reward", True)]:
        accs = [session(model, data, device, shuffled=shuf, seed=s) for s in range(3)]
        a = np.mean(accs, axis=0)
        n = len(a) // 4
        print(f"\n{tag}:")
        for q in range(4):
            print(f"  trials {q*n:>4}-{(q+1)*n:>4}: {a[q*n:(q+1)*n].mean()*100:5.1f}%")
        print(f"  overall: {a.mean()*100:.1f}%")
