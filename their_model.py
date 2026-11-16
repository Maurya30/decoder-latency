"""The UC Davis baseline decoder, rebuilt from their published code so it can run in this project.

Source: github.com/Neuroprosthetics-Lab/nejm-brain-to-text, model_training/rnn_model.py,
data_augmentations.py and evaluate_model_helpers.py (Card et al. 2024).

What happens to the rows, in order:
  1. smoothing: each row becomes a weighted average of itself and 4 rows either side
  2. a 512 x 512 input layer per recording day (the signals drift from day to day), softsign
  3. the newest 14 smoothed rows (280 ms) flattened into one vector, a new one every 4 rows (80 ms)
  4. a 5-layer GRU with 768 units, which carries a memory from step to step
  5. a linear layer to 41 phoneme scores

The weights are theirs, not trained here. `python -I import_their_weights.py` turns their
published checkpoint into data/pretrained/their-gru.pt, and load_their_gru() reads that.
Without that file the model is built with RANDOM weights: a step takes just as long, so
the timing is still real, but the output is noise.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from data import N_CLASSES, N_FEATURES, list_sessions

WEIGHTS = Path(__file__).parent / "data" / "pretrained" / "their-gru.pt"


def smoothing_kernel(std=2.0, cutoff=0.01):
    """Their Gaussian smoothing weights: 9 numbers, for rows t-4 to t+4.

    Same arithmetic as their gauss_smooth: a Gaussian with a standard deviation of 2 rows,
    the tails below 0.01 dropped, then scaled so the weights add up to 1.
    """
    offsets = torch.arange(-8, 9, dtype=torch.float32)
    weights = torch.exp(-0.5 * (offsets / std) ** 2)
    weights = weights / weights.sum()
    weights = weights[weights > cutoff]
    return weights / weights.sum()


class TheirGRU(nn.Module):
    patch = 14    # smoothed rows per step: patch_size in their config
    hop = 4       # new rows between steps: patch_stride in their config
    window = 22   # raw rows one step needs: 14, plus 8 more because smoothing looks 4 rows each way

    def __init__(self, sessions, n_units=768, n_layers=5):
        super().__init__()
        self.sessions = list(sessions)   # recording days, in the order of the day layers
        self.day = 0                     # which day's input layer to use
        self.n_units, self.n_layers = n_units, n_layers
        self.register_buffer("kernel", smoothing_kernel(), persistent=False)
        self.day_weights = nn.ParameterList(nn.Parameter(torch.eye(N_FEATURES)) for _ in self.sessions)
        self.day_biases = nn.ParameterList(nn.Parameter(torch.zeros(1, N_FEATURES)) for _ in self.sessions)
        self.gru = nn.GRU(N_FEATURES * self.patch, n_units, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(n_units, N_CLASSES)
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, n_units)))

    def use_day(self, day):
        """Pick the input layer that was trained for this recording day."""
        self.day = self.sessions.index(day)

    def _front_end(self, x):
        """Raw rows [batch, n, 512] -> smoothed, day-corrected rows [batch, n - 8, 512]."""
        x = x.unfold(1, len(self.kernel), 1) @ self.kernel   # every 9 neighbouring rows -> one weighted average
        return F.softsign(x @ self.day_weights[self.day] + self.day_biases[self.day])

    def _start_state(self, batch):
        return self.h0.expand(self.n_layers, batch, self.n_units).contiguous()

    def forward(self, x):
        """Whole sentence at once: [batch, time, 512] -> [batch, n_steps, 41]."""
        x = self._front_end(x)
        x = x.unfold(1, self.patch, self.hop)              # [batch, steps, 512, 14]
        x = x.permute(0, 1, 3, 2).flatten(start_dim=2)     # [batch, steps, 14 * 512]
        output, _ = self.gru(x, self._start_state(len(x)))
        return self.out(output)

    def out_len(self, n_rows):
        return (n_rows - self.window) // self.hop + 1

    def step(self, rows, state=None):
        """One live step: the newest 22 rows in, one row of scores out, memory carried in `state`."""
        x = self._front_end(rows).flatten(start_dim=1)[:, None]   # [1, 1, 14 * 512]
        if state is None:
            state = self._start_state(len(x))
        output, state = self.gru(x, state)
        return self.out(output), state


def load_their_gru(device="cpu"):
    """Returns (model, info). info is None if the published weights are not there and the weights are random."""
    if not WEIGHTS.exists():
        return TheirGRU(list_sessions()).to(device).eval(), None
    info = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = TheirGRU(info["sessions"])
    model.load_state_dict(info.pop("state"))
    return model.to(device).eval(), info
