"""How small can the CNN get before the accuracy falls apart?

    python sizes.py                # train every width three times (about 30 minutes), then build the table
    python sizes.py --table-only   # rebuild the table from models already trained

Each width gets the README's recipe (8 days, noise 0.5) with one change: how long to
train is no longer fixed at 12 epochs. A narrow model needs more epochs than a wide one,
so a fixed number is unfair to one or the other. Instead every 10th training sentence is
held back, and the weights kept are from the epoch that did best on those. A run stops
once 30 epochs pass without a better held-back score, or at 100 epochs. (30 is on
purpose long: training with CTC loss starts with a flat stretch where the score does
not move, and a shorter wait cuts narrow models off before they get going.) The
validation sentences never choose anything.

Each width is trained three times from different random starting points, because one
run on under 1,000 sentences moves by a few points on its own. Every trained model is
then converted to 8-bit (quantize.py), and both versions are scored on the same 267
validation sentences.

Width is the number of channels in the two hidden layers. The README's CNN is 256.
"""

import argparse
import datetime
import json
import subprocess
import sys

import charts
from data import N_CLASSES, N_FEATURES
from model import ROOT, RUNS, shown
from quantize import KERNEL, compare, quantize_file

WIDTHS = [256, 128, 64, 32, 16]
SEEDS = [0, 1, 2]
RECIPE = ["--model", "cnn", "--days", "8", "--noise", "0.5", "--holdout", "0.1", "--patience", "30", "--epochs", "100"]
FOLDER = RUNS / "sizes"
RESULTS = RUNS / "sizes.json"


def multiply_adds(width):
    """Multiplications in one live step (a count, not a measurement).

    Returns (kept, redone). One step is 4 new rows. The first layer has 7 output columns
    under the 19-row window, and only 2 of them are new each step. `kept` is when the
    other 5 are kept from the last step; `redone` is when all 7 are worked out again.
    """
    conv1_column = width * KERNEL * N_FEATURES
    rest = width * KERNEL * width + N_CLASSES * width   # conv2 once, then the head
    return 2 * conv1_column + rest, 7 * conv1_column + rest


def working_bytes(width):
    """RAM the Rust decoder needs while it runs: the size of its `Decoder` struct.

    7 input rows of 512 bytes, 7 first-layer columns and 1 second-layer column of `width`
    bytes, 41 four-byte scores and 4 bytes of bookkeeping. rust_check.py checks this
    against the size the Rust compiler reports.
    """
    return KERNEL * N_FEATURES + KERNEL * width + width + 4 * N_CLASSES + 4


def train_all(widths, seeds):
    """Train every (width, seed) that is not on disk yet. Output goes to runs/sizes/train.log."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    with open(FOLDER / "train.log", "a") as log:
        for width in widths:
            for seed in seeds:
                out = FOLDER / f"cnn_w{width}_s{seed}.pt"
                if out.exists():
                    continue
                print(f"training width {width}, seed {seed} ...", flush=True)
                log.write(f"=== width {width}, seed {seed}\n")
                log.flush()
                subprocess.run([sys.executable, "train.py", *RECIPE, "--width", str(width), "--seed", str(seed),
                                "--out", str(out)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def measure(widths, seeds):
    """Convert every trained model to 8-bit and score both versions. Returns the table as a dict."""
    table = {}
    for width in widths:
        runs = []
        for seed in seeds:
            path = FOLDER / f"cnn_w{width}_s{seed}.pt"
            if not path.exists():
                continue
            model, q, _, info, facts = quantize_file(path)
            result = compare(model, q, info["days"])
            runs.append({"seed": seed, "epochs": info["epochs"], "float_per": result["float_per"], "int8_per": result["int8_per"],
                         "same_best_phoneme": result["same_best_phoneme"]})
            print(f"  width {width:>3} seed {seed}: float {100 * result['float_per']:.1f}%, "
                  f"8-bit {100 * result['int8_per']:.1f}%", flush=True)
        if not runs:
            continue
        kept, redone = multiply_adds(width)
        table[str(width)] = {
            "parameters": facts["parameters"], "float_bytes": facts["float_bytes"],
            "int8_bytes": facts["int8_bytes"], "sentences": result["sentences"],
            "ram_bytes": working_bytes(width),
            "multiply_adds_kept": kept, "multiply_adds_redone": redone, "runs": runs,
        }
    return table


def spread(runs, key):
    """(mean, lowest, highest) of one error rate over the seeds, in percent."""
    values = [100 * run[key] for run in runs]
    return sum(values) / len(values), min(values), max(values)


def show(table):
    print(f"\n{'width':>5}{'parameters':>12}{'8-bit file':>13}{'RAM in use':>12}{'mult-adds/step':>16}"
          f"{'float, mean (range)':>24}{'8-bit, mean (range)':>24}")
    for width, row in table.items():
        cells = ["{:.1f}% ({:.1f} to {:.1f})".format(*spread(row["runs"], key)) for key in ("float_per", "int8_per")]
        print(f"{width:>5}{row['parameters']:>12,}{row['int8_bytes']:>11,} B{row['ram_bytes']:>10,} B{row['multiply_adds_kept']:>16,}"
              f"{cells[0]:>24}{cells[1]:>24}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--table-only", action="store_true", help="do not train, only measure what is on disk")
    args = parser.parse_args()

    if not args.table_only:
        train_all(args.widths, args.seeds)
    table = measure(args.widths, args.seeds)
    show(table)
    RESULTS.write_text(json.dumps({"measured": datetime.date.today().isoformat(), "recipe": " ".join(RECIPE),
                                   "widths": table}, indent=2))
    charts.size_chart(table, RUNS / "sizes.png")
    print(f"\nsaved {shown(RESULTS)} and {shown(RUNS / 'sizes.png')}")


if __name__ == "__main__":
    main()
