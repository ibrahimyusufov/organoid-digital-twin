import numpy as np
from pathlib import Path

MASK_PATH = Path("clone_data/live_channels.npy")
SHARE_THRESHOLD = 0.01   # keep channels carrying >=1% of all spikes
BIN_SECONDS = 0.05

if __name__ == "__main__":
    d = np.load("clone_data/binned_activity.npy")
    share = d.sum(axis=0) / d.sum()
    live = np.flatnonzero(share >= SHARE_THRESHOLD)
    print(f"live channels ({len(live)}): {live.tolist()}")
    for c in live:
        print(f"  ch {c:>2}: {share[c]*100:5.2f}% of spikes, "
              f"{d[:,c].mean()/BIN_SECONDS:.2f} Hz mean")
    print(f"retained: {d[:, live].sum()/d.sum()*100:.1f}%")
    np.save(MASK_PATH, live)
