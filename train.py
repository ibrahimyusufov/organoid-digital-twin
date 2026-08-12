from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_PATH = Path("clone_data/binned_activity.npy")
MODEL_PATH = Path("clone_data/clone_model.pt")

NUM_ELECTRODES = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
WINDOW = 200
BATCH_SIZE = 64
NUM_STEPS = 3000
LEARNING_RATE = 1e-3
LOG_EVERY = 100
SAVE_EVERY = 500


class SpikeCloneLSTM(nn.Module):
    def __init__(self, num_electrodes=NUM_ELECTRODES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(num_electrodes, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, num_electrodes)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        log_rate = self.head(out)
        return log_rate, hidden


def sample_batch(data, window: int = WINDOW, batch_size: int = BATCH_SIZE):
    max_start = data.shape[0] - window - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    batch = np.stack([data[s:s + window + 1] for s in starts])
    return torch.from_numpy(batch)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load(DATA_PATH)
    print(f"Loaded activity matrix: {data.shape}")

    model = SpikeCloneLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, NUM_STEPS + 1):
        sequence = sample_batch(data).to(device)
        inputs = torch.log1p(sequence[:, :-1, :])
        targets = sequence[:, 1:, :]

        log_rate, _ = model(inputs)
        loss = loss_fn(log_rate, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0:
            print(f"step {step}/{NUM_STEPS} loss {loss.item():.4f}")

        if step % SAVE_EVERY == 0 or step == NUM_STEPS:
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
