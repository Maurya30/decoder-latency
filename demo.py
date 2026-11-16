"""A page that shows the decoders running live, for demos.

    python demo.py            # then open http://localhost:8765

Pick a validation sentence and press Play. The server replays the recording at real
speed, one 20 ms frame at a time, through the same pieces the rest of the project uses:
frames.py loses frames on purpose and the receiver patches the holes, each decoder runs
one step every 80 ms, and stream.time_one_step times every step. The page (demo.html)
draws what arrives, what each decoder makes of it, and how long every step took.

Everything is a replay of the public UC Davis recordings on this computer. Nothing here
is live from a person.
"""

import argparse
import errno
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from data import BIN_MS, PHONEMES, load_split
from frames import Receiver, send, unpack_frame
from stream import build, summarise, time_one_step
from their_model import WEIGHTS
from train import edit_distance

PAGE = Path(__file__).parent / "demo.html"
NAMES = {"cnn": "This CNN", "their-gru": "UC Davis GRU, their weights"}
FILLS = ("ignore", "zeros", "hold")

MODELS = {}      # name -> model, filled in by main()
SENTENCES = []   # (day, features, labels, text)
LOCK = threading.Lock()   # one replay at a time, so two replays never share the processor
newest_replay = 0         # goes up with every Play press; an older replay stops when it sees a newer one


def for_drawing(values):
    """256 threshold-crossing values -> whole numbers 0..255 for the page to draw (about -1 to 4 covers them)."""
    return np.clip((values + 1) / 5 * 255, 0, 255).astype(int).tolist()


def replay(sentence, loss, burst, fill, seed):
    """Play one sentence at real speed. Yields (event name, details) for the page."""
    day, features, labels, text = sentence
    rows = features.numpy()
    arrived = {}   # sequence number -> frame, for the frames that make it across
    for frame in send(rows, loss=loss, burst=burst, rng=np.random.default_rng(seed)):
        arrived[unpack_frame(frame)[0]] = frame
    receiver = Receiver(fill)

    runs = {}
    for name, model in MODELS.items():
        if hasattr(model, "use_day"):   # only the UC Davis model has a separate input layer per day
            model.use_day(day)
        runs[name] = {"buffer": deque(maxlen=model.window), "seen": 0, "state": None,
                      "previous": None, "guess": [], "times": []}

    truth = labels.tolist()
    yield "start", {
        "rows": len(rows), "seconds": len(rows) * BIN_MS / 1000, "sentence": text,
        "truth": [PHONEMES[i] for i in truth],
        "models": [{"name": name, "label": NAMES[name], "window": model.window, "hop": model.hop,
                    "budget_ms": model.hop * BIN_MS} for name, model in MODELS.items()],
    }

    start = time.perf_counter()
    for i in range(len(rows)):
        wait = start + i * BIN_MS / 1000 - time.perf_counter()   # frame i is due i x 20 ms after the start
        if wait > 0:
            time.sleep(wait)
        if i not in arrived:
            yield "row", {"i": i, "lost": True}
            continue
        handed = receiver.push(arrived[i])   # the new row, plus any rows made up to patch a hole before it
        yield "row", {"i": i, "lost": False, "values": for_drawing(rows[i, :256])}
        for row in handed:
            row = torch.tensor(row)
            for name, model in MODELS.items():
                run = runs[name]
                run["buffer"].append(row)
                run["seen"] += 1
                if run["seen"] >= model.window and (run["seen"] - model.window) % model.hop == 0:
                    window = torch.stack(tuple(run["buffer"]))[None]
                    scores, run["state"], ms = time_one_step(model, window, run["state"], "cpu")
                    best = int(scores[0, -1].argmax())
                    is_new = best != run["previous"] and best != 0   # the greedy CTC rule: no repeats, no blanks
                    run["previous"] = best
                    if is_new:
                        run["guess"].append(best)
                    run["times"].append(ms)
                    yield "step", {"i": i, "model": name, "ms": round(ms, 3),
                                   "phoneme": PHONEMES[best] if is_new else None}

    results = {}
    for name, run in runs.items():
        budget = MODELS[name].hop * BIN_MS
        stats = summarise(run["times"]) if run["times"] else {"typical": 0, "p99": 0, "worst": 0}
        results[name] = {"wrong": edit_distance(run["guess"], truth), "of": len(truth), "steps": len(run["times"]),
                         "late": sum(t > budget for t in run["times"]), **stats}
    yield "end", {"results": results, "lost": len(rows) - len(arrived), "noticed": receiver.lost}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/sentences":
            listing = [{"id": i, "day": day, "text": text, "seconds": round(len(features) * BIN_MS / 1000, 1)}
                       for i, (day, features, _, text) in enumerate(SENTENCES)]
            self.reply(json.dumps(listing).encode(), "application/json")
        elif url.path == "/stream":
            self.stream(parse_qs(url.query))
        else:
            self.send_error(404)

    def reply(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream(self, query):
        """Send one replay to the page as a stream of events (server-sent events)."""
        global newest_replay

        def asked(name, default):
            return query.get(name, [default])[0]

        try:
            sentence = SENTENCES[int(asked("id", "0"))]
            loss = min(max(float(asked("loss", "0")), 0.0), 0.5)
            burst = min(max(int(asked("burst", "1")), 1), 50)
            fill = asked("fill", "zeros")
            seed = int(asked("seed", "0"))
            if fill not in FILLS:
                raise ValueError(fill)
        except (ValueError, IndexError):
            self.send_error(400)
            return
        newest_replay += 1
        mine = newest_replay
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with LOCK:
            try:
                for name, details in replay(sentence, loss, burst, fill, seed):
                    if mine != newest_replay:   # Play was pressed again: let the new replay take over
                        break
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(details)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass   # the page was closed mid-replay

    def log_message(self, *args):
        pass   # keep the terminal quiet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:   # claim the port first, so a clash shows up before the slow loading
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)   # this computer only
    except OSError as problem:
        if problem.errno != errno.EADDRINUSE:
            raise
        raise SystemExit(f"Port {args.port} is already in use, so the demo is probably running already: open "
                         f"http://localhost:{args.port}\nTo start a second copy:  python demo.py --port {args.port + 1}")

    print("loading the models and the sentences...")
    cnn, days, _ = build("cnn")
    MODELS["cnn"] = cnn
    if WEIGHTS.exists():
        MODELS["their-gru"] = build("their-gru")[0]
    else:
        print("the UC Davis weights are not here, so the page will show the CNN on its own (see the README)")
    for day in days:
        SENTENCES.extend((day, features, labels, text) for features, labels, text in load_split([day], "val"))

    print(f"{len(SENTENCES)} sentences, {len(MODELS)} decoder(s). Open http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
