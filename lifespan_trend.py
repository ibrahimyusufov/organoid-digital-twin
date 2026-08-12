from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_events, load_metadata

BIN_SIZE = "30min"
OUTPUT_PATH = Path("figures/lifespan_trend.png")


def compute_binned_rate(events, bin_size: str = BIN_SIZE):
    return events.set_index("time_of_event")["electrode"].resample(bin_size).count()


def plot_lifespan_trend(spike_rate, dead_date, output_path: Path = OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rolling = spike_rate.rolling(window=6, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spike_rate.index, spike_rate.values, linewidth=0.6, alpha=0.4, label=f"Spikes per {BIN_SIZE}")
    ax.plot(rolling.index, rolling.values, linewidth=2, label="Rolling trend")
    ax.axvline(dead_date, color="red", linestyle="--", label="Recorded death")
    ax.set_title("fs437 Organoid: Activity Trend Across Lifespan")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Spikes per {BIN_SIZE}")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    events = load_events()
    metadata = load_metadata()
    dead_date = pd.to_datetime(metadata["Dead Date"].iloc[0])

    spike_rate = compute_binned_rate(events)
    plot_lifespan_trend(spike_rate, dead_date)

    first_day = spike_rate[spike_rate.index < spike_rate.index.min() + pd.Timedelta(days=1)]
    last_day = spike_rate[spike_rate.index > spike_rate.index.max() - pd.Timedelta(days=1)]
    print(f"Mean spikes/{BIN_SIZE} in first 24h: {first_day.mean():.1f}")
    print(f"Mean spikes/{BIN_SIZE} in last 24h: {last_day.mean():.1f}")
