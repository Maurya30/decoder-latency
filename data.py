"""Load the public UC Davis speech-BCI recordings.

Data: Card et al. 2024, "An accurate and rapidly calibrating speech neuroprosthesis",
Dryad doi:10.5061/dryad.dncjsxm85. One participant (T15), 256 electrodes.

Each trial is one sentence the participant tried to say:
  features  [T, 512]  one row per 20 ms; 256 threshold-crossing counts, then 256
                      spike-band powers, already normalised by the dataset authors
  labels    [L]       the sentence as phoneme ids, in order, with NO timestamps
"""

from pathlib import Path

import h5py
import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data" / "hdf5_data_final"

N_FEATURES = 512   # 256 electrodes x 2 features
BIN_MS = 20        # one row of features covers 20 ms
N_CLASSES = 41     # id 0 = CTC blank, 1-39 = phonemes, 40 = silence between words

PHONEMES = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    "|",
]


def list_sessions():
    """Recording days that have both a training and a validation file, oldest first."""
    days = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    return [d for d in days if (DATA_DIR / d / "data_train.hdf5").exists()
            and (DATA_DIR / d / "data_val.hdf5").exists()]


def load_split(sessions, split):
    """Read every trial of one split ("train" or "val") for the given days into memory."""
    trials = []
    for day in sessions:
        with h5py.File(DATA_DIR / day / f"data_{split}.hdf5", "r") as f:
            for key in f:
                trial = f[key]
                features = torch.from_numpy(trial["input_features"][:])
                n_labels = int(trial.attrs["seq_len"])
                labels = torch.from_numpy(trial["seq_class_ids"][:n_labels].astype(np.int64))
                trials.append((features, labels, str(trial.attrs["sentence_label"])))
    return trials


def collate(batch):
    """Stack trials of different lengths into one padded batch, remembering true lengths."""
    features, labels, _ = zip(*batch)
    feature_lens = torch.tensor([len(x) for x in features])
    label_lens = torch.tensor([len(y) for y in labels])
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)  # [B, T, 512]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)      # [B, L]
    return features, feature_lens, labels, label_lens


if __name__ == "__main__":
    days = list_sessions()
    print(len(days), "recording days with train + val, from", days[0], "to", days[-1])
    trials = load_split(days[-1:], "val")
    features, labels, sentence = trials[0]
    print("one trial from", days[-1])
    print("  sentence:", sentence)
    print("  features:", tuple(features.shape), f"= {features.shape[0] * BIN_MS / 1000:.1f} s of 20 ms rows")
    print("  labels:  ", " ".join(PHONEMES[i] for i in labels.tolist()))
