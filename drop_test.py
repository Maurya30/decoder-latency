"""What happens to the decoder when frames go missing on the way to it?

    python drop_test.py                              # your trained CNN, scattered single-frame losses
    python drop_test.py --burst 10                   # losses come in runs of 10 frames (200 ms holes)
    python drop_test.py --model their-gru --burst 10 # the same test on the UC Davis model

Every validation sentence is sent as numbered frames (frames.py), some frames are
lost on purpose, the receiver does one of three things about it, and the decoder
runs on whatever comes out. The same frames are lost for all three, so the
comparison is fair.
"""

import argparse
import json

import numpy as np
import torch

import charts
from data import load_split
from frames import receive, send
from model import MODELS, RUNS, shown
from stream import build
from their_model import WEIGHTS
from train import decode, edit_distance

LOSS_RATES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
FILLS = ["ignore", "zeros", "hold"]


def error_rate_with_loss(model, by_day, loss, burst, fill, device, seed):
    """Phoneme error rate over all trials at one loss rate, plus how many frames were lost and noticed."""
    rng = np.random.default_rng(seed)
    errors = total = sent = arrived = noticed = 0
    for day, trials in by_day:
        if hasattr(model, "use_day"):   # only the UC Davis model has a separate input layer per day
            model.use_day(day)
        for features, labels, _ in trials:
            frames = list(send(features.numpy(), loss=loss, burst=burst, rng=rng))
            rows, receiver = receive(frames, fill)
            sent += len(features)
            arrived += len(frames)
            noticed += receiver.lost
            guess = decode(model, torch.from_numpy(rows), device)
            truth = labels.tolist()
            errors += edit_distance(guess, truth)
            total += len(truth)
    return errors / total, sent - arrived, noticed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=[*MODELS, "their-gru"])
    parser.add_argument("--burst", type=int, default=1, help="frames lost in a row each time")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model == "their-gru" and not WEIGHTS.exists():
        raise SystemExit("the UC Davis weights are not here: see the README, then run  python -I import_their_weights.py")
    model, days, _ = build(args.model, args.device)
    by_day = [(day, load_split([day], "val")) for day in days]
    n_trials = sum(len(trials) for _, trials in by_day)
    print(f"{args.model} on {n_trials} validation sentences, losses in runs of {args.burst}")
    print(f"{'frames lost':>12}  " + "  ".join(f"{fill:>8}" for fill in FILLS) + "   lost / noticed by the receiver")

    results = {"model": args.model, "burst": args.burst, "loss": LOSS_RATES, "per": {fill: [] for fill in FILLS}}
    for loss in LOSS_RATES:
        cells = []
        for fill in FILLS:
            per, lost, noticed = error_rate_with_loss(model, by_day, loss, args.burst, fill, args.device, args.seed)
            results["per"][fill].append(per)
            cells.append(f"{100 * per:7.1f}%")
        print(f"{100 * loss:11.1f}%  " + "  ".join(cells) + f"   {lost} / {noticed}", flush=True)
    print("(a loss at the very end of a sentence cannot be noticed: no later frame arrives to show the gap)")

    burst = "" if args.burst == 1 else f"_burst{args.burst}"
    name = "drop_test" + ("" if args.model == "cnn" else f"_{args.model}") + burst
    (RUNS / f"{name}.json").write_text(json.dumps(results, indent=2))
    charts.drop_chart(results, RUNS / f"{name}.png")
    print(f"saved {shown(RUNS / (name + '.json'))} and {shown(RUNS / (name + '.png'))}")

    # Once both models have been through the same test, draw them side by side on one scale.
    both = [RUNS / f"drop_test{burst}.json", RUNS / f"drop_test_their-gru{burst}.json"]
    if all(path.exists() for path in both):
        charts.drop_chart([json.loads(path.read_text()) for path in both], RUNS / f"drop_compare{burst}.png")
        print(f"saved {shown(RUNS / ('drop_compare' + burst + '.png'))}")


if __name__ == "__main__":
    main()
