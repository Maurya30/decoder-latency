"""Turn the UC Davis published checkpoint into a plain weights file this project can load.

    python -I import_their_weights.py

First download t15_pretrained_rnn_baseline.zip from the Dryad page (see the README) and
unzip it into data/pretrained/. This script then reads

    data/pretrained/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint   their saved model
    data/pretrained/t15_pretrained_rnn_baseline/checkpoint/args.yaml         their settings

and writes data/pretrained/their-gru.pt: the model's weights, plus the list of recording
days in the order their day-specific input layers use.

A checkpoint is a pickle file, and opening a pickle can run code. So it is opened with
weights_only=True, which only lets plain tensors and containers through. Their file also
holds one plain NumPy number (the validation loss), so exactly the three NumPy types
that number needs are allowed and nothing else. The script imports nothing of ours
(-I tells Python to ignore the current folder).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

# What a pickled NumPy float is made of. The first entry is named the way NumPy 1 named it,
# because that is how their file refers to it.
NUMPY_NUMBER = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, np.dtypes.Float64DType]

HERE = Path(__file__).parent
SOURCE = HERE / "data" / "pretrained" / "t15_pretrained_rnn_baseline"
OUT = HERE / "data" / "pretrained" / "their-gru.pt"


def read_sessions(path):
    """The recording days listed under `sessions:` in their settings file, in order."""
    sessions, inside = [], False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line == "sessions:":
            inside = True
        elif inside and line.startswith("- "):
            sessions.append(line[2:])
        elif inside:
            break
    return sessions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE, help="the unzipped t15_pretrained_rnn_baseline folder")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    sessions = read_sessions(args.source / "checkpoint" / "args.yaml")
    with torch.serialization.safe_globals(NUMPY_NUMBER):
        checkpoint = torch.load(args.source / "checkpoint" / "best_checkpoint", map_location="cpu", weights_only=True)

    state = {}
    for key, tensor in checkpoint["model_state_dict"].items():
        key = key.replace("module.", "").replace("_orig_mod.", "")   # prefixes their training setup adds
        state[key] = tensor.float()
    n_day_layers = sum(key.startswith("day_weights.") for key in state)
    if n_day_layers != len(sessions):
        raise SystemExit(f"{n_day_layers} day layers in the checkpoint but {len(sessions)} days in args.yaml")

    their_val_per = float(checkpoint["val_PER"])   # what their own training run measured for this checkpoint
    torch.save({"state": state, "sessions": sessions, "their_val_per": their_val_per}, args.out)
    n_weights = sum(t.numel() for t in state.values())
    print(f"{len(state)} tensors, {n_weights:,} weights, {len(sessions)} recording days")
    print(f"their own validation phoneme error rate for this checkpoint, over all days: {100 * their_val_per:.1f}%")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
