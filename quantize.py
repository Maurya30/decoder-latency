"""Turn a trained CNN's 32-bit float weights into 8-bit integers, and measure what that costs.

    python quantize.py                           # runs/cnn.pt -> runs/cnn_int8.bin
    python quantize.py --model runs/cnn_w64.pt   # any CNN that train.py saved

A microcontroller has little memory and often no fast float maths. The usual answer is
to store every weight as one signed byte and do the whole forward pass in integers.

How a float becomes a byte: pick a step size (a "scale") and store round(value / scale).

  input rows      one scale for all 512 features. The dataset clips its features at 10,
                  so 10 maps to 127.
  conv weights    one scale per output channel: that channel's largest weight maps to 127.
  layer outputs   nothing is negative after the ReLU, so these are stored as 0..255.
                  The scale is set by what the layer puts out on the TRAINING sentences.
                  The validation sentences are never used to set anything.
  head weights    one scale for all 41 classes, so the 41 integer scores can be compared
                  as they are and the best one picked without converting anything back.

Between layers a 32-bit sum has to come back down to one byte. That is a multiply by a
whole number and a right shift (fixed-point maths), so after the input rows have been
scaled there is no float anywhere in the model.

int_forward() below is the reference. The Rust decoder in decoder-rs/ reads the same
.bin file and must give exactly the same integers.
"""

import argparse
import datetime
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import N_CLASSES, N_FEATURES, load_split
from model import RUNS, SmallCNN, load_model, shown
from train import decode, edit_distance, greedy_decode

RESULTS = RUNS / "int8.json"
MAGIC = b"DLQ1"
KERNEL = 7
IN_CLIP = 10.0   # the dataset authors clip their normalised features at 10


def quantize_rows(rows, inv_scale):
    """Float rows [T, 512] -> int8. Nearest whole number, halves away from zero, kept within +-127.

    Written out step by step in 32-bit floats so the Rust version can do the identical sums.
    """
    v = np.asarray(rows, dtype=np.float32) * np.float32(inv_scale)
    v = np.where(v >= 0, v + np.float32(0.5), v - np.float32(0.5))
    return np.clip(np.trunc(v), -127, 127).astype(np.int8)


def output_ranges(model, trials, percentile):
    """How big each hidden layer's outputs get on these sentences.

    Returns (a1, a2): the values that `percentile` percent of conv1's and conv2's outputs
    stay under. Those two numbers become 255 in the integer model; anything bigger is cut
    off at 255.
    """
    seen1, seen2 = [], []
    with torch.no_grad():
        for rows, _, _ in trials:
            if len(rows) < model.window:
                continue
            h1 = torch.relu(model.conv1(rows.T[None]))
            h2 = torch.relu(model.conv2(h1))
            seen1.append(h1.flatten()[::17].numpy())   # a regular sample is plenty to find a percentile
            seen2.append(h2.flatten()[::5].numpy())
    return (float(np.percentile(np.concatenate(seen1), percentile)),
            float(np.percentile(np.concatenate(seen2), percentile)))


def to_int8(model, a1, a2):
    """Build the integer model from the float one. a1 and a2 are the layer outputs that map to 255."""
    scale_in = IN_CLIP / 127
    scale_1, scale_2 = a1 / 255, a2 / 255
    q = {"width": model.width, "in_inv_scale": np.float32(127 / IN_CLIP)}
    hidden = [(1, model.conv1, scale_in, 127, scale_1), (2, model.conv2, scale_1, 255, scale_2)]
    for i, conv, scale_x, biggest_x, scale_y in hidden:
        w = conv.weight.detach().double().numpy()                       # [out, in, 7]
        scale_w = np.maximum(np.abs(w).max(axis=(1, 2)), 1e-12) / 127   # one per output channel
        q[f"w{i}"] = np.round(w / scale_w[:, None, None]).astype(np.int8)
        # A sum of (weight byte * input byte) is in units of scale_x * scale_w. The bias joins it in those units.
        bias = np.round(conv.bias.detach().double().numpy() / (scale_x * scale_w))
        # The sum then has to be rescaled to the output's 0..255 units: multiply by `ratio`.
        # ratio is a fraction, so it is stored as a whole number m with ratio ~= m / 2**shift.
        ratio = scale_x * scale_w / scale_y
        shift = int(min(40, np.floor(31 - np.log2(ratio.max()) - 1e-9)))   # the biggest shift that keeps m inside 32 bits
        m = np.round(ratio * 2.0 ** shift)
        assert np.abs(bias).max() < 2 ** 31 and m.max() < 2 ** 31
        q[f"b{i}"], q[f"m{i}"], q[f"shift{i}"] = bias.astype(np.int32), m.astype(np.int32), shift
        # Proof that a 32-bit sum cannot overflow, whatever the input: take every input byte at its largest.
        worst = np.abs(q[f"w{i}"].astype(np.int64)).sum(axis=(1, 2)) * biggest_x + np.abs(bias)
        assert worst.max() < 2 ** 31, "a 32-bit sum could overflow"

    w = model.head.weight.detach().double().numpy()[:, :, 0]            # [41, width]
    scale_w = max(np.abs(w).max(), 1e-12) / 127                         # one for the whole layer
    q["w3"] = np.round(w / scale_w).astype(np.int8)
    bias = np.round(model.head.bias.detach().double().numpy() / (scale_2 * scale_w))
    worst = np.abs(q["w3"].astype(np.int64)).sum(axis=1) * 255 + np.abs(bias)
    assert worst.max() < 2 ** 31, "a 32-bit score could overflow"
    q["b3"] = bias.astype(np.int32)
    return q


def int_conv(x, w, b, stride):
    """An exact integer convolution. x is [T, in] whole numbers, w is [out, in, K], b is [out].

    It runs in 64-bit floats so the fast matrix code can be used. Every product and every
    running sum here is a whole number far below 2**53, and 64-bit floats hold all of those
    exactly, so nothing is rounded. check.py compares it with a slow pure-integer version.
    """
    as_float = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
    out = F.conv1d(as_float(x.T)[None], as_float(w), as_float(b), stride=stride)
    return out[0].T.numpy().astype(np.int64)   # [T_out, out]


def requantize(sums, m, shift):
    """ReLU, then bring 32-bit sums down to 0..255: multiply by m, add half, shift right, cap at 255."""
    sums = np.maximum(sums, 0)
    return np.minimum((sums * m.astype(np.int64) + (1 << (shift - 1))) >> shift, 255).astype(np.uint8)


def int_forward(q, rows_q):
    """A whole sentence through the integer model. rows_q is int8 [T, 512]; returns scores [T_out, 41]."""
    h = requantize(int_conv(rows_q, q["w1"], q["b1"], 2), q["m1"], q["shift1"])
    h = requantize(int_conv(h, q["w2"], q["b2"], 2), q["m2"], q["shift2"])
    return int_conv(h, q["w3"][:, :, None], q["b3"], 1)


def int_decode(q, rows):
    """Float rows [T, 512] in, phoneme guess out, through the integer model."""
    if len(rows) < SmallCNN.window:
        return []
    scores = int_forward(q, quantize_rows(rows.numpy(), q["in_inv_scale"]))
    return greedy_decode(torch.from_numpy(scores), len(scores))


def save_int8(q, path):
    """Write the integer model as one flat file: the bytes a microcontroller would keep in flash.

    Layout, little-endian. Weights are ordered [output channel][tap][input] so that one
    output's sum walks straight through memory.
      "DLQ1", then u32 features, width, classes, kernel, shift1, shift2, then f32 input scale (127 / 10)
      w1 i8 [width][7][512]    b1 i32 [width]    m1 i32 [width]
      w2 i8 [width][7][width]  b2 i32 [width]    m2 i32 [width]
      w3 i8 [41][width]        b3 i32 [41]
    """
    parts = [MAGIC, struct.pack("<6If", N_FEATURES, q["width"], N_CLASSES, KERNEL,
                                q["shift1"], q["shift2"], q["in_inv_scale"])]
    for i in (1, 2):
        parts += [q[f"w{i}"].transpose(0, 2, 1).tobytes(), q[f"b{i}"].astype("<i4").tobytes(),
                  q[f"m{i}"].astype("<i4").tobytes()]
    parts += [q["w3"].tobytes(), q["b3"].astype("<i4").tobytes()]
    Path(path).write_bytes(b"".join(parts))


def load_int8(path):
    """Read a file written by save_int8 back into the arrays int_forward uses."""
    blob = Path(path).read_bytes()
    assert blob[:4] == MAGIC, f"{path} is not an integer model file"
    features, width, classes, kernel, shift1, shift2, inv_scale = struct.unpack_from("<6If", blob, 4)
    assert (features, classes, kernel) == (N_FEATURES, N_CLASSES, KERNEL)
    q = {"width": width, "in_inv_scale": np.float32(inv_scale), "shift1": shift1, "shift2": shift2}
    at = 4 + struct.calcsize("<6If")

    def take(dtype, *shape):
        nonlocal at
        array = np.frombuffer(blob, dtype=dtype, count=int(np.prod(shape)), offset=at).reshape(shape)
        at += array.nbytes
        return array

    for i, n_in in ((1, features), (2, width)):
        q[f"w{i}"] = take(np.int8, width, kernel, n_in).transpose(0, 2, 1)   # back to [out, in, 7]
        q[f"b{i}"], q[f"m{i}"] = take("<i4", width), take("<i4", width)
    q["w3"], q["b3"] = take(np.int8, classes, width), take("<i4", classes)
    assert at == len(blob), "the file is longer than its header says"
    return q


def compare(model, q, days):
    """Float model against integer model on the validation sentences of `days`.

    Returns a dict with both error rates, overall and per day, and how often the two
    models pick the same phoneme at the same output step.
    """
    per_day = {"float": [], "int8": []}
    wrong = {"float": 0, "int8": 0}
    phonemes = steps = same = sentences = 0
    for day in days:
        trials = load_split([day], "val")
        day_wrong = {"float": 0, "int8": 0}
        day_phonemes = sum(len(labels) for _, labels, _ in trials)
        for rows, labels, _ in trials:
            day_wrong["float"] += edit_distance(decode(model, rows, "cpu"), labels.tolist())
            day_wrong["int8"] += edit_distance(int_decode(q, rows), labels.tolist())
            if len(rows) >= model.window:
                with torch.no_grad():
                    best_float = model(rows[None])[0].argmax(dim=-1).numpy()
                best_int = int_forward(q, quantize_rows(rows.numpy(), q["in_inv_scale"])).argmax(axis=-1)
                same += int((best_float == best_int).sum())
                steps += len(best_int)
        for kind in wrong:
            wrong[kind] += day_wrong[kind]
            per_day[kind].append(day_wrong[kind] / day_phonemes)
        phonemes += day_phonemes
        sentences += len(trials)
    return {"sentences": sentences, "days": days,
            "float_per": wrong["float"] / phonemes, "int8_per": wrong["int8"] / phonemes,
            "float_per_day": per_day["float"], "int8_per_day": per_day["int8"],
            "steps": steps, "same_best_phoneme": same / steps}


def quantize_file(model_path, percentile=99.99, out=None):
    """Float model file in, integer model file out. Returns (model, q, out_path, facts)."""
    model, info = load_model(model_path)
    a1, a2 = output_ranges(model, load_split(info["days"], "train"), percentile)
    out = Path(out or Path(model_path).with_name(Path(model_path).stem + "_int8.bin"))
    save_int8(to_int8(model, a1, a2), out)
    q = load_int8(out)   # score the file itself: these are the bytes the Rust decoder reads
    n_params = sum(p.numel() for p in model.parameters())
    facts = {"width": model.width, "parameters": n_params, "float_bytes": 4 * n_params,
             "int8_bytes": out.stat().st_size, "percentile": percentile,
             "conv1_output_at_255": a1, "conv2_output_at_255": a2}
    return model, q, out, info, facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUNS / "cnn.pt"), help="a CNN saved by train.py")
    parser.add_argument("--percentile", type=float, default=99.99,
                        help="the share of each hidden layer's training outputs that must fit under 255")
    parser.add_argument("--out", default=None, help="where to write (default: next to the model, _int8.bin)")
    args = parser.parse_args()

    model, q, out, info, facts = quantize_file(args.model, args.percentile, args.out)
    print(f"{shown(args.model)}: {facts['parameters']:,} parameters, width {facts['width']}")
    print(f"  as 32-bit floats  {facts['float_bytes']:>9,} bytes")
    print(f"  as 8-bit integers {facts['int8_bytes']:>9,} bytes   saved {shown(out)}")

    result = compare(model, q, info["days"])
    print(f"\n{result['sentences']} validation sentences from {len(result['days'])} days. Phoneme error rate, lower is better.")
    print(f"{'day':<16}{'float':>9}{'8-bit':>9}")
    for day, a, b in zip(result["days"], result["float_per_day"], result["int8_per_day"]):
        print(f"{day:<16}{100 * a:>8.1f}%{100 * b:>8.1f}%")
    print(f"{'all days':<16}{100 * result['float_per']:>8.1f}%{100 * result['int8_per']:>8.1f}%")
    print(f"\nthe two pick the same phoneme on {100 * result['same_best_phoneme']:.1f}% of {result['steps']:,} output steps")

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results[Path(args.model).stem] = {"measured": datetime.date.today().isoformat(), **facts, **result}
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"saved {shown(RESULTS)}")


if __name__ == "__main__":
    main()
