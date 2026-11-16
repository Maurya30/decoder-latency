"""Run a decoder live: one 20 ms row at a time, timing every step.

    python stream.py                           # your trained CNN, on the CPU and on the Mac's GPU (MPS)
    python stream.py --paced                   # rows arrive every 20 ms, like a real stream
    python stream.py --models cnn their-gru    # add the UC Davis model for comparison

What one step is: a new row arrives and goes into a ring buffer that holds the newest
`window` rows. Every `hop` rows, the model turns the buffer into one row of phoneme
scores. The clock runs from "window is ready" to "scores are back on the CPU".

"their-gru" is the UC Davis baseline with their published weights, or with random
weights if those have not been imported (same timing, but the output is noise).
See their_model.py.
"""

import argparse
import json
import time
from collections import deque

import numpy as np
import torch

import charts
from data import BIN_MS, list_sessions, load_split
from model import MODELS, RUNS, load_model, shown
from their_model import load_their_gru

RESULTS = RUNS / "latency.json"


def time_one_step(model, window, state, device):
    """Run one live step and time it.

    window is [1, model.window, 512] and lives on the CPU. `state` is whatever the
    model handed back last step (None for the CNN, the hidden state for a GRU).
    Returns (scores, state, elapsed_ms) with scores back on the CPU.
    """
    start = time.perf_counter()
    with torch.no_grad():   # no training here, so skip the gradient bookkeeping
        scores, state = model.step(window.to(device), state)
    # The GPU works in the background. Copying the scores back to the CPU makes us wait
    # until it has really finished, so the clock cannot stop early.
    scores = scores.cpu()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return scores, state, elapsed_ms


def summarise(times_ms):
    """Boil a list of step times down to three numbers.

    Returns {"typical": ..., "p99": ..., "worst": ...}, all in milliseconds.
    typical is the median. p99 is the time that 99 steps in 100 stay under.
    """
    times = np.asarray(times_ms)
    return {
        "typical": float(np.median(times)),
        "p99": float(np.percentile(times, 99)),
        "worst": float(times.max()),
    }


def replay(model, rows, device, paced=False, max_steps=None):
    """Feed rows to the model one at a time, the way they would arrive live.

    Returns (scores [n_steps, 41], times_ms). With paced=True each row is held back
    until its real arrival time, so the processor idles between steps like it would
    on a real stream.
    """
    buffer = deque(maxlen=model.window)   # ring buffer: only ever the newest `window` rows
    state, scores, times = None, [], []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        if paced:
            wait = start + i * BIN_MS / 1000 - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        buffer.append(row)
        seen = i + 1
        if seen >= model.window and (seen - model.window) % model.hop == 0:
            window = torch.stack(tuple(buffer))[None]
            out, state, elapsed_ms = time_one_step(model, window, state, device)
            scores.append(out[0, -1])
            times.append(elapsed_ms)
            if max_steps and len(times) >= max_steps:
                break
    return torch.stack(scores), times


def check_matches_offline(model, rows):
    """The live loop must give the same scores as running the whole sentence at once."""
    model.to("cpu")
    live, _ = replay(model, rows, "cpu")
    with torch.no_grad():
        offline = model(rows[None])[0]
    if live.shape != offline.shape:
        raise SystemExit(f"live loop gave {len(live)} steps, whole-sentence run gave {len(offline)}: "
                         "check the model's window and hop")
    worst = (live - offline).abs().max().item()
    if worst > 1e-3:
        raise SystemExit(f"live and whole-sentence scores differ by {worst:.4f}: check window and hop")
    return worst


def build(name, device="cpu"):
    """Returns (model, days, note). days are the recording days the model gets tested on."""
    if name == "their-gru":
        model, info = load_their_gru(device)
        note = "UC Davis published weights" if info else "UC Davis design, random weights (timing only)"
        return model, list_sessions()[-8:], note
    path = RUNS / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"no trained model at {shown(path)}: run  python train.py --model {name}  first")
    model, info = load_model(path, device)
    return model, info["days"], f"trained here, {100 * info['val_per']:.1f}% phoneme error rate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["cnn"], choices=[*MODELS, "their-gru"])
    parser.add_argument("--devices", nargs="+", default=None, help="cpu and/or mps (default: both if available)")
    parser.add_argument("--steps", type=int, default=None, help="timed steps per run (default 2000, or 500 with --paced)")
    parser.add_argument("--warmup", type=int, default=50, help="first steps to throw away")
    parser.add_argument("--paced", action="store_true", help="hold each row back until its real arrival time")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    parser.add_argument("--fresh", action="store_true", help="forget the timing runs saved earlier")
    args = parser.parse_args()

    devices = args.devices or (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"])
    steps = args.steps or (500 if args.paced else 2000)
    mode = "paced" if args.paced else "back to back"
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"{steps} timed steps per run after {args.warmup} warm-up steps, rows {mode}, "
          f"{torch.get_num_threads()} CPU threads")

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() and not args.fresh else {}
    for name in args.models:
        model, days, note = build(name)
        if hasattr(model, "use_day"):   # only the UC Davis model has a separate input layer per day
            model.use_day(days[-1])
        trials = load_split(days[-1:], "val")
        check_matches_offline(model, trials[0][0])
        rows = torch.cat([features for features, _, _ in trials])
        needed = model.window + (steps + args.warmup) * model.hop
        while len(rows) < needed:   # loop the recording if one day is not long enough
            rows = torch.cat([rows, rows])
        budget_ms = model.hop * BIN_MS
        print(f"\n{name}: {note}")
        print(f"  sees {model.window} rows ({model.window * BIN_MS} ms) per step, one step every "
              f"{model.hop} rows ({budget_ms} ms). Live loop matches the whole-sentence run.")
        for device in devices:
            model.to(device)
            _, times = replay(model, rows, device, paced=args.paced, max_steps=steps + args.warmup)
            times = times[args.warmup:]
            stats = summarise(times)
            late = sum(t > budget_ms for t in times)
            print(f"  {device:>3}: typical {stats['typical']:7.3f} ms   p99 {stats['p99']:7.3f} ms   "
                  f"worst {stats['worst']:7.3f} ms   over the {budget_ms} ms budget: {late} of {len(times)}")
            results[f"{name} / {device} / {mode}"] = {
                "model": name, "device": device, "mode": mode, "steps": len(times),
                "budget_ms": budget_ms, "over_budget": late, "note": note, **stats,
            }

    RUNS.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2))
    charts.latency_chart(results, RUNS / "latency.png")
    print(f"\nsaved {shown(RESULTS)} and {shown(RUNS / 'latency.png')}")


if __name__ == "__main__":
    main()
