"""Does the Rust decoder give the same integers as the Python one, and how long does a step take?

    python rust_check.py              # runs/cnn_int8.bin: every validation sentence, then the timing
    python rust_check.py --sizes      # the smaller models from sizes.py too (seed 0 of each width)
    python rust_check.py --no-timing  # only the comparison

Needs Rust (https://rustup.rs) and the .bin files that quantize.py and sizes.py write.

The comparison: every validation sentence goes through decoder-rs one row at a time, and
all 41 scores at every output step must equal quantize.py's int_forward exactly. Not
"close": the same integers. Inside the Rust test bench every step is also worked out a
second way (the whole window from scratch) and the two must agree.

The timing is on this laptop. It says nothing about a microcontroller.
"""

import argparse
import datetime
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

from data import BIN_MS, N_CLASSES, list_sessions, load_split
from model import ROOT, RUNS, SmallCNN, load_model, shown
from quantize import int_forward, load_int8, quantize_rows
from sizes import WIDTHS, working_bytes
from stream import replay, summarise
from train import edit_distance, greedy_decode

CRATE = ROOT / "decoder-rs"
RESULTS = RUNS / "rust.json"
DAYS = 8   # the same 8 most recent days as everything else here


def build():
    """Compile decoder-rs and return the path of its test bench."""
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(cargo).exists():
        raise SystemExit("Rust is not installed: see https://rustup.rs")
    subprocess.run([cargo, "build", "--release", "--quiet"], cwd=CRATE, check=True)
    return CRATE / "target" / "release" / "host"


def as_bytes(rows):
    """Float rows [T, 512] the way the test bench reads them: a row count, then the rows."""
    return struct.pack("<I", len(rows)) + np.ascontiguousarray(rows, dtype="<f4").tobytes()


def compare(host, model_path, days):
    """Every validation sentence through the Rust decoder, checked score by score against Python."""
    q = load_int8(model_path)
    bench = subprocess.Popen([host, "check", model_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    sentences = steps = differing = wrong = phonemes = 0
    for day in days:
        for rows, labels, _ in load_split([day], "val"):
            rows = rows.numpy()
            bench.stdin.write(as_bytes(rows))
            bench.stdin.flush()
            n_steps, = struct.unpack("<I", bench.stdout.read(4))
            rust = np.frombuffer(bench.stdout.read(4 * N_CLASSES * n_steps), dtype="<i4").reshape(n_steps, N_CLASSES)
            if len(rows) >= SmallCNN.window:
                python = int_forward(q, quantize_rows(rows, q["in_inv_scale"]))
            else:
                python = np.zeros((0, N_CLASSES), dtype=np.int64)
            if rust.shape != python.shape:
                raise SystemExit(f"Rust gave {len(rust)} steps for a sentence, Python gave {len(python)}")
            differing += int((rust != python).any(axis=1).sum())
            guess = greedy_decode(torch.from_numpy(rust.astype(np.int64)), n_steps) if n_steps else []
            wrong += edit_distance(guess, labels.tolist())
            phonemes += len(labels)
            steps += n_steps
            sentences += 1
    bench.stdin.write(struct.pack("<I", 0))
    bench.stdin.close()
    if bench.wait() != 0:
        raise SystemExit("the Rust test bench stopped with an error (see above)")
    return {"sentences": sentences, "steps": steps, "steps_that_differ": differing, "rust_per": wrong / phonemes}


def timing(host, model_path, rows, paced, steps=None, warmup=50):
    """Step times from the Rust test bench. Returns ({mode: typical/p99/worst}, the bench's raw answer)."""
    steps = steps or (500 if paced else 2000)   # the same counts stream.py uses
    needed = SmallCNN.window + (steps + warmup) * SmallCNN.hop
    while len(rows) < needed:   # loop the recording if one day is not long enough
        rows = np.concatenate([rows, rows])
    done = subprocess.run([host, "time", model_path, str(steps), str(warmup), "1" if paced else "0"],
                          input=as_bytes(rows[:needed]), stdout=subprocess.PIPE, check=True)
    raw = json.loads(done.stdout)
    mode = "paced" if paced else "back to back"
    budget_ms = SmallCNN.hop * BIN_MS
    found = {}
    for how in ("kept", "redone"):
        times = raw.pop(f"{how}_ms")
        found[f"{how} / {mode}"] = {"steps": len(times), "budget_ms": budget_ms,
                                    "over_budget": sum(t > budget_ms for t in times), **summarise(times)}
    return found, raw


def pytorch_timing(rows, paced, steps=None, warmup=50):
    """The float CNN in PyTorch on the same rows, timed the way stream.py does it.

    It is here so the Rust numbers have a reference from the same day and the same laptop.
    """
    model, _ = load_model(RUNS / "cnn.pt")
    steps = steps or (500 if paced else 2000)
    needed = SmallCNN.window + (steps + warmup) * SmallCNN.hop
    while len(rows) < needed:
        rows = np.concatenate([rows, rows])
    _, times = replay(model, torch.from_numpy(rows[:needed]), "cpu", paced=paced, max_steps=steps + warmup)
    times = times[warmup:]
    budget_ms = SmallCNN.hop * BIN_MS
    mode = "paced" if paced else "back to back"
    return {f"pytorch float / {mode}": {"steps": len(times), "budget_ms": budget_ms, "threads": torch.get_num_threads(),
                                        "over_budget": sum(t > budget_ms for t in times), **summarise(times)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", action="store_true", help="also the smaller models from sizes.py")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()

    models = {256: RUNS / "cnn_int8.bin"}
    if args.sizes:
        models.update({w: RUNS / "sizes" / f"cnn_w{w}_s0_int8.bin" for w in WIDTHS if w != 256})
    for path in models.values():
        if not path.exists():
            raise SystemExit(f"{shown(path)} is not there: run  python quantize.py  (and  python sizes.py  for --sizes)")

    host = build()
    days = list_sessions()[-DAYS:]
    timing_rows = torch.cat([rows for rows, _, _ in load_split(days[-1:], "val")]).numpy()

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for width, path in models.items():
        print(f"\nwidth {width}: {shown(path)}")
        found = compare(host, str(path), days)
        verdict = "identical" if found["steps_that_differ"] == 0 else f"DIFFERENT on {found['steps_that_differ']} steps"
        print(f"  {found['sentences']} sentences, {found['steps']:,} output steps, 41 scores each: {verdict}")
        print(f"  phoneme error rate from the Rust decoder's output: {100 * found['rust_per']:.1f}%")
        entry = {"measured": datetime.date.today().isoformat(), "model": str(shown(path)), **found}
        if not args.no_timing:
            entry["timing"] = {}
            for paced in (False, True):
                modes, raw = timing(host, str(path), timing_rows, paced)
                if width == 256:   # the README's CNN: time the PyTorch version of it alongside
                    modes.update(pytorch_timing(timing_rows, paced))
                entry["timing"].update(modes)
                entry["state_bytes"], entry["model_bytes"] = raw["decoder_bytes"], raw["model_bytes"]
                assert raw["decoder_bytes"] == working_bytes(width), "sizes.py has the working memory wrong"
                for mode, t in modes.items():
                    print(f"  {mode:<29} typical {t['typical']:7.3f} ms   p99 {t['p99']:7.3f} ms   "
                          f"worst {t['worst']:7.3f} ms   over the {t['budget_ms']} ms budget: {t['over_budget']} of {t['steps']}")
            print(f"  working memory (the Decoder struct): {entry['state_bytes']:,} bytes")
        elif str(width) in results:   # keep an earlier timing when only the comparison was rerun
            entry = {**results[str(width)], **entry}
        results[str(width)] = entry
        if found["steps_that_differ"]:
            raise SystemExit("the Rust decoder does not match the Python one")

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {shown(RESULTS)}")


if __name__ == "__main__":
    main()
