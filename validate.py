from pathlib import Path

import numpy as np
import torch

from model import SpikeCloneLSTM

DATA_PATH = Path("clone_data/binned_activity.npy")
MODEL_PATH = Path("clone_data/clone_model.pt")

WINDOW = 200
NUM_SAMPLES = 200


def load_model(device):
    model = SpikeCloneLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def teacher_forced_predicted_rate(model, data, device, window: int = WINDOW, num_samples: int = NUM_SAMPLES):
    max_start = data.shape[0] - window - 1
    starts = np.random.randint(0, max_start, size=num_samples)
    batch = np.stack([data[s:s + window + 1] for s in starts])
    sequence = torch.from_numpy(batch).to(device)

    inputs = torch.log1p(sequence[:, :-1, :])
    targets = sequence[:, 1:, :]

    with torch.no_grad():
        log_rate, _ = model(inputs)
        predicted_rate = torch.exp(log_rate)

    return predicted_rate.mean().item(), targets.float().mean().item()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = np.load(DATA_PATH)
    model = load_model(device)

    predicted_mean, real_mean = teacher_forced_predicted_rate(model, data, device)
    print(f"Teacher-forced predicted mean rate (per bin, per electrode): {predicted_mean:.4f}")
    print(f"Real mean count (per bin, per electrode): {real_mean:.4f}")
