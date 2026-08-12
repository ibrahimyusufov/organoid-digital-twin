from pathlib import Path

import numpy as np
import pandas as pd

from load_data import load_events

BIN_SIZE = "50ms"
NUM_ELECTRODES = 32
TRAIN_START_HOURS = 0
TRAIN_END_HOURS = 48
OUTPUT_PATH = Path("clone_data/binned_activity.npy")


def bin_events(events, bin_size: str = BIN_SIZE):
    counts = (
        events
        .assign(time_bin=events["time_of_event"].dt.floor(bin_size))
        .groupby(["time_bin", "electrode"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(NUM_ELECTRODES), fill_value=0)
    )
    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq=bin_size)
    return counts.reindex(full_index, fill_value=0)


if __name__ == "__main__":
    events = load_events()

    start_time = events["time_of_event"].min() + pd.Timedelta(hours=TRAIN_START_HOURS)
    end_time = events["time_of_event"].min() + pd.Timedelta(hours=TRAIN_END_HOURS)
    window = events[(events["time_of_event"] >= start_time) & (events["time_of_event"] < end_time)]

    binned = bin_events(window)
    print(f"Binned shape (time_steps, electrodes): {binned.shape}")
    print(f"Time range used for training: {start_time} to {end_time}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, binned.values.astype("float32"))
    print(f"Saved binned activity to {OUTPUT_PATH}")
