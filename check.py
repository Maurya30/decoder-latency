"""Checks that the key pieces behave. Run it after changing any of them.

    python check.py

None of these need the dataset. rust_check.py is the one that compares the Rust
decoder with the Python one on the real validation sentences.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import frames
import quantize
import stream
from data import N_CLASSES, N_FEATURES
from model import SmallCNN


def cnn_layers():
    model = SmallCNN()
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
    assert len(convs) == 3, f"expected 3 Conv1d layers, found {len(convs)}"
    assert convs[0].in_channels == N_FEATURES, "the first layer should take 512 features in"
    assert convs[-1].out_channels == N_CLASSES, "the last layer should give 41 scores out"
    assert [c.stride[0] for c in convs] == [2, 2, 1], "strides should be 2, 2, 1"


def cnn_forward():
    model = SmallCNN().eval()
    with torch.no_grad():
        scores = model(torch.randn(2, 100, N_FEATURES))
    assert scores.shape == (2, 21, N_CLASSES), f"100 rows in should give shape (2, 21, 41), got {tuple(scores.shape)}"
    x = torch.randn(1, 40, N_FEATURES)
    with torch.no_grad():   # without a ReLU the model is a straight line, and f(x) + f(-x) = 2 f(0)
        straight = torch.allclose(model(x) + model(-x), 2 * model(torch.zeros_like(x)), atol=1e-3)
    assert not straight, "the output is a straight-line function of the input: is the ReLU in there?"


def cnn_out_len():
    model = SmallCNN().eval()
    for n in (19, 20, 23, 100, 239, 729, 1427):
        with torch.no_grad():
            real = model(torch.zeros(1, n, N_FEATURES)).shape[1]
        assert model.out_len(n) == real, f"out_len({n}) says {model.out_len(n)}, the model really gives {real}"


def cnn_window_hop():
    model = SmallCNN()
    assert model.window is not None and model.hop is not None, "window and hop are still None"
    assert model.out_len(model.window) == 1, f"{model.window} rows should give exactly 1 output step"
    assert model.out_len(model.window - 1) == 0, f"{model.window - 1} rows already give an output step: window is too big"
    assert model.out_len(model.window + model.hop) == 2, "window + hop rows should give exactly 2 output steps"
    assert model.out_len(model.window + model.hop - 1) == 1, "one row fewer than window + hop should still give 1 step"


class _SlowModel(torch.nn.Module):
    """Pretends to be a decoder that takes 20 ms a step, to check the clock."""

    def step(self, rows, state=None):
        time.sleep(0.02)
        return torch.zeros(1, 1, N_CLASSES, device=rows.device), "kept"


def timing():
    out = stream.time_one_step(_SlowModel(), torch.zeros(1, 4, N_FEATURES), None, "cpu")
    assert isinstance(out, tuple) and len(out) == 3, "return (scores, state, elapsed_ms)"
    scores, state, elapsed_ms = out
    assert state == "kept", "hand back the state that model.step returned"
    assert scores.device.type == "cpu", "the scores should be back on the CPU"
    assert 19 < elapsed_ms < 60, f"a 20 ms step was timed as {elapsed_ms:.4f}: is it in milliseconds?"


def summary():
    stats = stream.summarise([float(i) for i in range(1, 101)])
    assert set(stats) >= {"typical", "p99", "worst"}, "return a dict with typical, p99 and worst"
    assert abs(stats["typical"] - 50.5) < 0.01, f"typical of 1..100 should be 50.5, got {stats['typical']}"
    assert 98.9 < stats["p99"] <= 100, f"p99 of 1..100 should be about 99, got {stats['p99']}"
    assert stats["worst"] == 100, f"worst of 1..100 should be 100, got {stats['worst']}"


def gap_check():
    receiver = frames.Receiver()
    assert receiver.missing_before(0) == 0, "the first frame (seq 0) arriving first means nothing is missing"
    receiver.last_seq = 41
    assert receiver.missing_before(42) == 0, "42 right after 41: nothing missing"
    assert receiver.missing_before(43) == 1, "43 right after 41: one frame missing"
    assert receiver.missing_before(50) == 8, "50 right after 41: eight frames missing"


def _random_int8_model(width=16):
    """A small untrained CNN turned into integers, plus the float model it came from."""
    torch.manual_seed(0)
    model = SmallCNN(width).eval()
    return model, quantize.to_int8(model, 4.0, 4.0)


def int8_rounding():
    got = quantize.quantize_rows([[0.0, 0.03, 0.04, -0.04, 10.0, -10.0, 99.0, -99.0]], 12.7)
    assert got.dtype.name == "int8", "rows should come back as int8"
    assert got.tolist() == [[0, 0, 1, -1, 127, -127, 127, -127]], f"0.04 * 12.7 rounds to 1 and 10 maps to 127, got {got.tolist()}"
    assert quantize.quantize_rows([[0.5, -0.5, 1.5]], 1.0).tolist() == [[1, -1, 2]], "halves round away from zero"


def int8_exact_sums():
    _, q = _random_int8_model()
    rng = np.random.default_rng(0)
    x = rng.integers(-127, 128, size=(40, N_FEATURES)).astype(np.int8)
    fast = quantize.int_conv(x, q["w1"], q["b1"], 2)
    slow = np.empty_like(fast)   # the same sums in plain 64-bit integers, one output step at a time
    for t in range(len(slow)):
        window = x[2 * t: 2 * t + 7].astype(np.int64)   # [7, 512]
        slow[t] = np.einsum("oik,ki->o", q["w1"].astype(np.int64), window) + q["b1"]
    assert np.array_equal(fast, slow), "int_conv is not giving exact integer sums"
    sums = np.array([[-5, 0, 1, 2 ** 20, 2 ** 30]])
    got = quantize.requantize(sums, np.array([2 ** 20] * 5, dtype=np.int32), 30).tolist()
    assert got == [[0, 0, 0, 255, 255]], f"negatives should become 0 and big sums stop at 255, got {got}"
    half = quantize.requantize(np.array([[511, 512, 1536]]), np.array([2 ** 20] * 3, dtype=np.int32), 30).tolist()
    assert half == [[0, 1, 2]], f"the shift should round to nearest, halves up, got {half}"


def int8_file():
    _, q = _random_int8_model()
    with tempfile.TemporaryDirectory() as folder:
        path = f"{folder}/model.bin"
        quantize.save_int8(q, path)
        back = quantize.load_int8(path)
        size = Path(path).stat().st_size
    for key, value in q.items():
        assert np.array_equal(back[key], value), f"{key} changed between saving and loading"
    width = q["width"]
    expected = 32 + width * 7 * N_FEATURES + width * 7 * width + N_CLASSES * width + 4 * (4 * width + N_CLASSES)
    assert size == expected, f"the file should be {expected} bytes, it is {size}"


def int8_tracks_float():
    model, _ = _random_int8_model(32)
    torch.manual_seed(1)
    rows = torch.randn(200, N_FEATURES).clamp(-10, 10)
    a1, a2 = quantize.output_ranges(model, [(rows, None, None)], 100.0)
    q = quantize.to_int8(model, a1, a2)
    with torch.no_grad():
        real = model(rows[None])[0].numpy()
    ints = quantize.int_forward(q, quantize.quantize_rows(rows.numpy(), q["in_inv_scale"]))
    assert ints.shape == real.shape, f"integer scores have shape {ints.shape}, float scores {real.shape}"
    # One integer score step is worth (conv2's output scale) * (head weight scale) in float terms.
    step = (a2 / 255) * (np.abs(model.head.weight.detach().numpy()).max() / 127)
    worst = np.abs(ints * step - real).max()
    assert worst < 0.1 * (real.max() - real.min()), f"integer scores drift from the float ones by {worst:.3f}"


CHECKS = [
    ("model.py    1  the three CNN layers", cnn_layers),
    ("model.py    2  the forward pass", cnn_forward),
    ("model.py    3  out_len", cnn_out_len),
    ("model.py    4  window and hop", cnn_window_hop),
    ("stream.py   1  time_one_step", timing),
    ("stream.py   2  summarise", summary),
    ("frames.py   1  missing_before", gap_check),
    ("quantize.py 1  rows to 8-bit", int8_rounding),
    ("quantize.py 2  exact integer sums", int8_exact_sums),
    ("quantize.py 3  the model file", int8_file),
    ("quantize.py 4  8-bit tracks float", int8_tracks_float),
]


def main():
    done = 0
    for label, check in CHECKS:
        try:
            check()
            verdict = "works"
            done += 1
        except NotImplementedError:
            verdict = "not written yet"
        except AssertionError as problem:
            verdict = f"not right yet: {problem}"
        except Exception as problem:   # a typo or a shape error in the new lines
            verdict = f"crashed: {type(problem).__name__}: {problem}"
        print(f"  {label:<38} {verdict}")
    print(f"{done} of {len(CHECKS)} done")


if __name__ == "__main__":
    main()
