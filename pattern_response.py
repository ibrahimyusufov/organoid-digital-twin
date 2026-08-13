from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_events, load_stimulations

BIN_S = 0.05
PRE_S, POST_S = 1.0, 2.0
LIVE = np.load("clone_data/live_channels.npy")
FIG = Path("figures/pattern_response.png")

PATTERNS = {
    "A (18,19,21,23)": (18, 19, 21, 23),
    "B (10,11,13,15)": (10, 11, 13, 15),
    "C (1,9,12,14)":   (1, 9, 12, 14),
}

def psth(ev_ns, ev_ch, stim_times, n_ch):
    """Per-channel spike counts in bins around each stim onset."""
    edges = np.arange(-PRE_S, POST_S + BIN_S, BIN_S)
    out = np.zeros((n_ch, len(edges) - 1))
    for st in stim_times:
        t = np.datetime64(st).astype("int64")
        lo = np.searchsorted(ev_ns, t - int(PRE_S * 1e9))
        hi = np.searchsorted(ev_ns, t + int(POST_S * 1e9))
        if hi <= lo:
            continue
        rel = (ev_ns[lo:hi] - t) / 1e9
        ch = ev_ch[lo:hi]
        for i, c in enumerate(LIVE):
            m = ch == c
            if m.any():
                out[i] += np.histogram(rel[m], bins=edges)[0]
    return edges, out

if __name__ == "__main__":
    ev = load_events().sort_values("time_of_event")
    st = load_stimulations()
    t0 = ev["time_of_event"].min()
    st = st[(st["time_of_stim"] >= t0) & (st["time_of_stim"] < t0 + pd.Timedelta(hours=48))]

    groups = st.groupby("time_of_stim")["electrode"].apply(lambda x: tuple(sorted(x)))
    ev_ns = ev["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    ev_ch = ev["electrode"].values

    results = {}
    for name, pat in PATTERNS.items():
        times = groups[groups == pat].index.values
        edges, counts = psth(ev_ns, ev_ch, times, len(LIVE))
        results[name] = (counts, len(times))
        pre = counts[:, edges[:-1] < 0].sum() / (PRE_S / BIN_S)
        post = counts[:, (edges[:-1] >= 0) & (edges[:-1] < 0.5)].sum() / (0.5 / BIN_S)
        print(f"{name:<18} n={len(times):>5}  pre {pre:8.1f}  post {post:8.1f}  "
              f"ratio {post/pre if pre else float('nan'):.2f}")

    print("\nper-channel response (first 0.5 s after onset, spikes per bin):")
    print("ch    " + "".join(f"{k.split()[0]:>10}" for k in PATTERNS))
    post_m = {}
    for name, (c, n) in results.items():
        post_m[name] = c[:, (edges[:-1] >= 0) & (edges[:-1] < 0.5)].sum(axis=1) / max(n, 1)
    for i, ch in enumerate(LIVE):
        print(f"{ch:<6}" + "".join(f"{post_m[k][i]:>10.3f}" for k in PATTERNS))

    vecs = np.array([post_m[k] for k in PATTERNS])
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    print("\ncosine similarity between response patterns:")
    keys = list(PATTERNS)
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  {keys[i].split()[0]} vs {keys[j].split()[0]}: {vn[i] @ vn[j]:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    ctr = (edges[:-1] + edges[1:]) / 2
    for a, (name, (c, n)) in zip(ax, results.items()):
        a.bar(ctr, c.sum(axis=0) / max(n, 1), width=BIN_S * 0.9)
        a.axvline(0, color="red", ls="--")
        a.set_title(f"{name}\nn={n}")
        a.set_xlabel("time from stim (s)")
    ax[0].set_ylabel("spikes per stim per bin")
    fig.tight_layout(); FIG.parent.mkdir(exist_ok=True); fig.savefig(FIG, dpi=150)
    print(f"\nsaved {FIG}")
