from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_data import load_events, load_stimulations

WINDOW_SECONDS = 2.0
BIN_SECONDS = 0.05
SAMPLE_SIZE = 5000
RANDOM_SEED = 0
OUTPUT_PATH = Path("figures/stimulation_response.png")


def build_psth(events, stim_times, window=WINDOW_SECONDS, bin_size=BIN_SECONDS):
    event_times_ns = events["time_of_event"].values.astype("datetime64[ns]").astype("int64")
    window_ns = int(window * 1e9)
    bin_edges = np.arange(-window, window + bin_size, bin_size)
    counts = np.zeros(len(bin_edges) - 1)

    for stim_time in stim_times:
        stim_ns = np.datetime64(stim_time).astype("int64")
        lo = np.searchsorted(event_times_ns, stim_ns - window_ns)
        hi = np.searchsorted(event_times_ns, stim_ns + window_ns)
        if hi <= lo:
            continue
        relative = (event_times_ns[lo:hi] - stim_ns) / 1e9
        hist, _ = np.histogram(relative, bins=bin_edges)
        counts += hist

    return bin_edges, counts


def plot_psth(bin_edges, counts, output_path: Path = OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, counts, width=BIN_SECONDS * 0.9)
    ax.axvline(0, color="red", linestyle="--", label="Stimulation onset")
    ax.set_title(f"fs437 Organoid: Spike Response Around Stimulation (n={SAMPLE_SIZE} stims)")
    ax.set_xlabel("Time relative to stimulation (s)")
    ax.set_ylabel("Total spike count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    events = load_events().sort_values("time_of_event")
    stims = load_stimulations()

    unique_stim_times = stims["time_of_stim"].drop_duplicates().sort_values()
    print(f"Unique stimulation timestamps: {len(unique_stim_times)}")

    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(
        unique_stim_times.values, size=min(SAMPLE_SIZE, len(unique_stim_times)), replace=False
    )

    bin_edges, counts = build_psth(events, sample)

    pre = counts[bin_edges[:-1] < 0].sum()
    post = counts[bin_edges[:-1] >= 0].sum()
    print(f"Total spikes in pre-stim window: {pre:.0f}")
    print(f"Total spikes in post-stim window: {post:.0f}")

    plot_psth(bin_edges, counts)
