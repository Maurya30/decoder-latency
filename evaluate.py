"""Phoneme error rate of each decoder on the same validation sentences, day by day.

    python evaluate.py

Compares the two models trained here (the one-layer baseline and the CNN) with the
UC Davis published model (their weights, see their_model.py). All of them get exactly
the same sentences: the validation split of the recording days the models here were
trained on.
"""

import argparse
import json
import re

from data import PHONEMES, load_split
from model import MODELS, RUNS, shown
from stream import build
from their_model import WEIGHTS
from train import decode, edit_distance

THEIR_LOG = WEIGHTS.parent / "t15_pretrained_rnn_baseline" / "training_log"


def their_logged_error_rates():
    """What UC Davis's own training log says for each day, at the step their checkpoint was saved.

    Returns {day: error rate}. It is the check that the model rebuilt in their_model.py
    behaves like the one they trained: the two columns should agree closely.
    """
    if not THEIR_LOG.exists():
        return {}
    lines = THEIR_LOG.read_text().splitlines()
    saved = max(i for i, line in enumerate(lines) if "Checkpointing model" in line)
    start = max(i for i in range(saved) if "Val batch" in lines[i])
    found = {}
    for line in lines[start:saved]:
        match = re.search(r"(t15\.\d{4}\.\d\d\.\d\d) val PER: ([\d.]+)", line)
        if match:
            found[match.group(1)] = float(match.group(2))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[*MODELS, "their-gru"], choices=[*MODELS, "their-gru"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if "their-gru" in args.models and not WEIGHTS.exists():
        raise SystemExit("the UC Davis weights are not here: see the README, then run  python -I import_their_weights.py")
    built = {name: build(name, args.device) for name in args.models}
    days = built[args.models[0]][1]   # test every model on the first model's days
    models = {name: model for name, (model, _, _) in built.items()}

    errors = {name: [] for name in models}   # phonemes wrong, per day
    totals, counts, example = [], [], None
    for day in days:
        trials = load_split([day], "val")
        totals.append(sum(len(labels) for _, labels, _ in trials))
        counts.append(len(trials))
        for name, model in models.items():
            if hasattr(model, "use_day"):   # only the UC Davis model has a separate input layer per day
                model.use_day(day)
            guesses = [decode(model, features, args.device) for features, _, _ in trials]
            errors[name].append(sum(edit_distance(g, labels.tolist()) for g, (_, labels, _) in zip(guesses, trials)))
            if day == days[0]:
                example = example or {"sentence": trials[0][2], "truth": trials[0][1].tolist(), "guess": {}}
                example["guess"][name] = guesses[0]

    their_log = their_logged_error_rates() if "their-gru" in models else {}
    print(f"{sum(counts)} validation sentences from {len(days)} days. Phoneme error rate, lower is better.")
    print(f"{'day':<16}{'sentences':>10}" + "".join(f"{name:>12}" for name in models)
          + ("   their own log, same model" if their_log else ""))
    for i, day in enumerate(days):
        print(f"{day:<16}{counts[i]:>10}" + "".join(f"{100 * errors[name][i] / totals[i]:>11.1f}%" for name in models)
              + (f"{100 * their_log[day]:>11.1f}%" if day in their_log else ""))
    overall = {name: sum(errors[name]) / sum(totals) for name in models}
    print(f"{'all ' + str(len(days)) + ' days':<16}{sum(counts):>10}" + "".join(f"{100 * overall[name]:>11.1f}%" for name in models))

    print("\nexample sentence:", example["sentence"])
    print(f"  {'truth':<10}", " ".join(PHONEMES[i] for i in example["truth"]))
    for name, guess in example["guess"].items():
        print(f"  {name:<10}", " ".join(PHONEMES[i] for i in guess))

    results = {"days": days, "sentences": counts, "overall": overall,
               "per_day": {name: [e / t for e, t in zip(errors[name], totals)] for name in models},
               "their_own_log": [their_log.get(day) for day in days]}
    (RUNS / "accuracy.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {shown(RUNS / 'accuracy.json')}")


if __name__ == "__main__":
    main()
