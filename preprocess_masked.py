import numpy as np
from pathlib import Path

SRC = Path("clone_data/binned_activity.npy")
MASK = Path("clone_data/live_channels.npy")
OUT = Path("clone_data/binned_activity_masked.npy")
BIN_SECONDS = 0.05

if __name__ == "__main__":
    d = np.load(SRC)
    live = np.load(MASK)
    masked = np.ascontiguousarray(d[:, live])
    print(f"{d.shape} -> {masked.shape}")
    print(f"mean rate: {masked.sum(axis=1).mean()/BIN_SECONDS:.2f} Hz")
    np.save(OUT, masked)
