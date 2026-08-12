from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import SpikeCloneLSTM

DATA_PATH = Path("clone_data/binned_activity.npy")
MODEL_PATH = Path("clone_data/clone_model.pt")
OUTPUT_PATH = Path("clone_data/generated_activity.npy")
FIGURE_PATH = Path("figures/ai_clone_generated_activity.png")

SEED_WINDOW = 200
RESEED_INTERVAL = 200
GENERATE_STEPS = 6000
BIN_SECONDS = 0.05
AGGREGATE_SECONDS = 1.0


def load_model(device):
    model = SpikeCloneLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def generate(model, data, steps, device, seed_window: int = SEED_WINDOW, reseed_interval: int = RESEED_INTERVAL):
    generated = np.zeros((steps, data.shape[1]), dtype="float32")

    t = 0
    with torch.no_grad():
        while t < steps:
            seed_start = np.random.randint(0, data.shape[0] - seed_window)
            seed = data[seed_start:seed_start + seed_window]
            seed_input = torch.log1p(torch.from_numpy(seed)).unsqueeze(0).to(device)

            log_rate_seq, hidden = model(seed_input)
            log_rate = log_rate_seq[:, -1, :]

            segment_steps = min(reseed_interval, steps - t)
            for _ in range(segment_steps):
                rate = torch.exp(log_rate).clamp(min=0).cpu()
                counts_cpu = torch.poisson(rate)
                generated[t] = counts_cpu[0].numpy()
                t += 1

                counts = counts_cpu.to(device)
                next_input = torch.log1p(counts).unsqueeze(1)
                log_rate_seq, hidden = model(next_input, hidden)
                log_rate = log_rate_seq[:, -1, :]

    return generated


def plot_generated(generated, output_path: Path = FIGURE_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_per_bin = generated.sum(axis=1)
    steps_per_second = int(AGGREGATE_SECONDS / BIN_SECONDS)
    trimmed = total_per_bin[: len(total_per_bin) - (len(total_per_bin) % steps_per_second)]
    per_second = trimmed.reshape(-1, steps_per_second).sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(per_second)), per_second, linewidth=1.0, color="tab:purple")
    ax.set_title("AI Clone: Generated Organoid Activity (Never-Dying Digital Twin)")
    ax.set_xlabel("Generated time (s)")
    ax.set_ylabel("Total spikes per second (all electrodes)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_PATH)
    model = load_model(device)
    generated = generate(model, data, GENERATE_STEPS, device)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, generated)
    print(f"Saved {GENERATE_STEPS} generated steps to {OUTPUT_PATH}")

    real = np.load(DATA_PATH)
    print(f"Real mean spikes/sec (all electrodes): {real.sum(axis=1).mean() / BIN_SECONDS:.2f}")
    print(f"Generated mean spikes/sec (all electrodes): {generated.sum(axis=1).mean() / BIN_SECONDS:.2f}")

    plot_generated(generated)
