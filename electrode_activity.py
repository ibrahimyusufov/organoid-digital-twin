from pathlib import Path

import matplotlib.pyplot as plt

from load_data import load_events

OUTPUT_PATH = Path("figures/electrode_activity.png")


def compute_electrode_counts(events):
    return events["electrode"].value_counts().sort_values(ascending=False)


def plot_electrode_counts(counts, output_path: Path = OUTPUT_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("fs437 Organoid: Total Spikes per Electrode")
    ax.set_xlabel("Electrode")
    ax.set_ylabel("Total Spike Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    events = load_events()
    counts = compute_electrode_counts(events)
    print(counts)
    plot_electrode_counts(counts)
