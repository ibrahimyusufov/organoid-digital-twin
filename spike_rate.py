from pathlib import Path

import matplotlib.pyplot as plt

from load_data import load_events

BIN_SIZE = "1min"
OUTPUT_PATH = Path("figures/spike_rate_over_time.png")


def compute_spike_rate(events, bin_size: str = BIN_SIZE):
    return events.set_index("time_of_event")["electrode"].resample(bin_size).count()


def plot_spike_rate(spike_rate, output_path: Path = OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spike_rate.index, spike_rate.values, linewidth=0.8)
    ax.set_title("fs437 Organoid: Spike Rate Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Spikes per {BIN_SIZE}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    events = load_events()
    spike_rate = compute_spike_rate(events)
    plot_spike_rate(spike_rate)
