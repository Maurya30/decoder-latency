"""Train a decoder on the UC Davis recordings and report phoneme error rate.

    python train.py --model cnn --days 8 --epochs 12 --noise 0.5
    python train.py --model linear --days 8 --epochs 5      # the no-context baseline

The labels have no timestamps, so the loss is CTC (connectionist temporal
classification): it scores every way the label sequence could line up with the
output steps and adds them up, so the model learns the alignment by itself.
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PHONEMES, collate, list_sessions, load_split
from model import MODELS, RUNS, save_model, shown


def greedy_decode(scores, n_steps):
    """Turn per-step scores into a phoneme sequence: best class per step,
    merge repeats, drop blanks. This is the standard greedy CTC decode."""
    best = scores[:n_steps].argmax(dim=-1).tolist()
    out, previous = [], None
    for c in best:
        if c != previous and c != 0:
            out.append(c)
        previous = c
    return out


def decode(model, rows, device):
    """Run the model over one sentence's rows [n, 512] and return its phoneme guess."""
    if len(rows) < model.window:   # too short for even one output step
        return []
    with torch.no_grad():
        scores = model(rows[None].to(device)).cpu()[0]
    return greedy_decode(scores, len(scores))


def edit_distance(a, b):
    """Fewest insertions, deletions and substitutions to turn sequence a into b."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        diagonal, row[0] = row[0], i
        for j, y in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (x != y))
    return row[-1]


def phoneme_error_rate(model, loader, device):
    """Share of phonemes the model gets wrong on a dataset (lower is better)."""
    model.eval()
    errors = total = 0
    with torch.no_grad():
        for features, feature_lens, labels, label_lens in loader:
            scores = model(features.to(device)).cpu()
            for i in range(len(scores)):
                guess = greedy_decode(scores[i], model.out_len(int(feature_lens[i])))
                truth = labels[i, : label_lens[i]].tolist()
                errors += edit_distance(guess, truth)
                total += len(truth)
    return errors / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="linear")
    parser.add_argument("--days", type=int, default=8, help="how many of the most recent recording days to use")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="random noise added to the training rows, so the model cannot just memorise them")
    parser.add_argument("--width", type=int, default=256, help="channels in the CNN's two hidden layers")
    parser.add_argument("--seed", type=int, default=0, help="starting point for the random numbers")
    parser.add_argument("--holdout", type=float, default=0.0,
                        help="share of the training sentences to keep back and never train on. They decide which "
                             "epoch's weights are kept, so the validation sentences never choose anything")
    parser.add_argument("--patience", type=int, default=30,
                        help="with --holdout: stop after this many epochs without a better held-back score")
    parser.add_argument("--out", default=None, help="where to save (default runs/<model>.pt)")
    args = parser.parse_args()
    out = args.out or RUNS / f"{args.model}.pt"
    sized = {"width": args.width} if args.model == "cnn" else {}

    torch.manual_seed(args.seed)
    days = list_sessions()[-args.days:]
    train = load_split(days, "train")
    val = load_split(days, "val")
    held = []
    if args.holdout > 0:   # every 10th training sentence, for --holdout 0.1
        every = round(1 / args.holdout)
        held, train = train[every - 1::every], [t for i, t in enumerate(train) if i % every != every - 1]
    print(f"{len(days)} days ({days[0]} to {days[-1]}): {len(train)} training sentences, {len(val)} validation"
          + (f", {len(held)} held back from training" if held else ""))

    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val, batch_size=args.batch, collate_fn=collate)
    held_loader = DataLoader(held, batch_size=args.batch, collate_fn=collate) if held else None

    model = MODELS[args.model](**sized).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, {n_params:,} parameters, on {args.device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = None   # with --holdout: the best epoch so far, as (held-back error rate, epoch, weights, validation error rate)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for features, feature_lens, labels, label_lens in train_loader:
            features = features.to(args.device)
            if args.noise > 0:
                features = features + args.noise * torch.randn_like(features)
            scores = model(features)                                   # [B, T_out, 41]
            log_probs = F.log_softmax(scores, dim=-1).transpose(0, 1)  # CTC wants [T_out, B, 41]
            out_lens = torch.tensor([model.out_len(int(n)) for n in feature_lens])
            # The CTC loss runs on the CPU; gradients still flow back to the model.
            loss = F.ctc_loss(log_probs.cpu(), labels, out_lens, label_lens, blank=0, zero_infinity=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # stops one bad batch wrecking the weights
            optimizer.step()
            running += loss.item()
        per = phoneme_error_rate(model, val_loader, args.device)
        line = f"epoch {epoch}: loss {running / len(train_loader):.3f}, phoneme error rate {100 * per:.1f}%"
        if held:
            held_per = phoneme_error_rate(model, held_loader, args.device)
            line += f", held back {100 * held_per:.1f}%"
            if best is None or held_per < best[0]:
                best = (held_per, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}, per)
        print(f"{line}, {time.time() - start:.0f} s", flush=True)
        if held and epoch - best[1] >= args.patience:
            break

    if held:
        held_per, epoch, state, per = best
        model.load_state_dict(state)
        print(f"kept epoch {epoch}: the best on the {len(held)} held-back training sentences ({100 * held_per:.1f}%). "
              f"Its validation error rate is {100 * per:.1f}%")
    save_model(model, args.model, out, days=days, epochs=epoch, val_per=per, seed=args.seed, **sized)
    print(f"saved {shown(out)}")

    features, labels, sentence = val[0]
    guess = decode(model.eval(), features, args.device)
    print("example sentence:", sentence)
    print("  truth:", " ".join(PHONEMES[i] for i in labels.tolist()))
    print("  model:", " ".join(PHONEMES[i] for i in guess))


if __name__ == "__main__":
    main()
