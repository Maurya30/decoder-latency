"""Decoders: brain features in, phoneme scores out.

Every model takes x with shape [batch, time, 512] and returns scores with shape
[batch, time_out, 41]. time_out can be smaller than time if the model strides.

Every model also says how it runs live, one step at a time:
  window   how many 20 ms rows one output step looks at
  hop      how many new rows arrive between one output step and the next
  step()   the newest `window` rows in, one row of phoneme scores out
"""

from pathlib import Path

import torch
from torch import nn

from data import N_CLASSES, N_FEATURES

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def shown(path):
    """A path the way it gets printed: relative to the project folder when it is inside it."""
    path = Path(path).resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


class LinearBaseline(nn.Module):
    """The simplest possible decoder. Scores each 20 ms row on its own, with no context.

    It exists to answer one question: how much does looking at neighbouring rows help?
    """

    window = 1
    hop = 1

    def __init__(self):
        super().__init__()
        self.out = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x):
        return self.out(x)

    def out_len(self, n_rows):
        """How many output steps come out for n_rows of input."""
        return n_rows

    def step(self, rows, state=None):
        """One live step. rows is [1, window, 512]; returns ([1, 1, 41], state)."""
        return self.forward(rows), state


class SmallCNN(nn.Module):
    """A small 1D CNN that slides along time.

    Spec:
      conv1   512 -> 256 channels, kernel 7, stride 2, then ReLU
      conv2   256 -> 256 channels, kernel 7, stride 2, then ReLU
      head    256 -> 41 channels, kernel 1 (one score per phoneme class)

    `width` is the 256 above. Smaller widths give the smaller models in sizes.py.
    """

    # How the model runs live. Both numbers follow from the kernels and strides below.
    window = 19   # input rows one output step can see: 7 + (7 - 1) * 2 = 19 rows = 380 ms
    hop = 4       # new rows between one output step and the next: stride 2 * stride 2 = 4 rows = 80 ms

    def __init__(self, width=256):
        super().__init__()
        self.width = width
        # The three layers, each an nn.Conv1d.
        self.conv1 = nn.Conv1d(N_FEATURES, width, kernel_size=7, stride=2)
        self.conv2 = nn.Conv1d(width, width, kernel_size=7, stride=2)
        self.head = nn.Conv1d(width, N_CLASSES, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)       # Conv1d wants [batch, features, time]
        # conv1, ReLU, conv2, ReLU, head.
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.head(x)
        return x.transpose(1, 2)    # back to [batch, time, classes]  
    
    

    def out_len(self, n_rows):
        """How many output steps come out for n_rows of input."""
        # A conv with kernel k and stride s turns n rows into (n - k) // s + 1. Once per layer.
        n_rows = (n_rows - 7) // 2 + 1
        n_rows = (n_rows - 7) // 2 + 1
        n_rows = (n_rows - 1) // 1 + 1
        return n_rows
    

    def step(self, rows, state=None):
        """One live step. rows is [1, window, 512]; returns ([1, 1, 41], state)."""
        return self.forward(rows), state


MODELS = {"linear": LinearBaseline, "cnn": SmallCNN}


def save_model(model, name, path, **info):
    """Save the weights plus what we need to reload and re-test the model later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "state": model.state_dict(), **info}, path)


def load_model(path, device="cpu"):
    """Load a model saved by train.py. Returns (model, info) with the model in eval mode."""
    info = torch.load(path, map_location="cpu", weights_only=True)
    sized = {"width": info["width"]} if "width" in info else {}   # only the narrower CNNs record a width
    model = MODELS[info["model"]](**sized)
    model.load_state_dict(info.pop("state"))
    return model.to(device).eval(), info
