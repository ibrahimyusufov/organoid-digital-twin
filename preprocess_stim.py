from pathlib import Path
import numpy as np
import pandas as pd
from load_data import load_events, load_stimulations

BIN_SIZE = "50ms"
BIN_S = 0.05
HOURS = 48
MASK = Path("clone_data/live_channels.npy")
OUT = Path("clone_data/stim_matrix.npy")

if __name__ == "__main__":
    live = np.load(MASK)
    ev = load_events()
    st = load_stimulations()

    t0 = ev["time_of_event"].min()
    t1 = t0 + pd.Timedelta(hours=HOURS)
    st = st[(st["time_of_stim"] >= t0) & (st["time_of_stim"] < t1)]
    st = st[st["electrode"].isin(live)]
    print(f"stim rows on live channels: {len(st)}")

    # match the spike matrix index exactly
    ev_w = ev[(ev["time_of_event"] >= t0) & (ev["time_of_event"] < t1)]
    bins = ev_w["time_of_event"].dt.floor(BIN_SIZE)
    index = pd.date_range(bins.min(), bins.max(), freq=BIN_SIZE)
    print(f"target index length: {len(index)}")

    col = {c: i for i, c in enumerate(live)}
    m = np.zeros((len(index), len(live)), dtype="float32")
    pos = index.get_indexer(st["time_of_stim"].dt.floor(BIN_SIZE))
    valid = pos >= 0
    rows = pos[valid]
    cols = st["electrode"].map(col).values[valid]
    amp = st["a1"].values[valid].astype("float32")
    np.add.at(m, (rows, cols), amp)

    print(f"stim matrix: {m.shape}")
    print(f"bins with any stim: {(m.sum(1) > 0).sum()} ({(m.sum(1) > 0).mean()*100:.3f}%)")
    print(f"amplitude range: {m[m>0].min():.2f} to {m.max():.2f}")
    np.save(OUT, m)
    print(f"saved {OUT}")
