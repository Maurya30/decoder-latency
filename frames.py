"""Numbered frames: how rows of features would travel from the implant side to the decoder.

    python frames.py        # small demo: frame size, a corrupted frame, a lost frame

One frame carries one 20 ms row:
    seq       uint32          goes up by 1 with every frame the sender makes
    payload   512 x float32   the row of features
    crc       uint32          CRC-32 of everything before it

This is a stand-in, not any real device's format. A real implant sends raw samples at a far
higher rate; these frames carry features that are already binned.
"""

import struct
import zlib

import numpy as np

from data import N_FEATURES

SEQ = struct.Struct("<I")   # little-endian unsigned 32-bit
CRC = struct.Struct("<I")
FRAME_BYTES = SEQ.size + N_FEATURES * 4 + CRC.size


def pack_frame(seq, row):
    """One row of features -> the bytes that go down the wire."""
    body = SEQ.pack(seq) + np.asarray(row, dtype="<f4").tobytes()
    return body + CRC.pack(zlib.crc32(body))


def unpack_frame(frame):
    """Bytes off the wire -> (seq, row), or None if the CRC says the frame is damaged."""
    body, (crc,) = frame[:-CRC.size], CRC.unpack(frame[-CRC.size:])
    if len(frame) != FRAME_BYTES or zlib.crc32(body) != crc:
        return None
    (seq,) = SEQ.unpack_from(body)
    return seq, np.frombuffer(body, dtype="<f4", offset=SEQ.size)


def send(rows, loss=0.0, burst=1, rng=None):
    """Yield one frame per row, losing some on purpose.

    loss is the share of frames that go missing overall. burst is how many go missing
    in a row each time (1 = scattered single frames, 10 = 200 ms holes).
    """
    rng = rng or np.random.default_rng(0)
    to_skip = 0
    for seq, row in enumerate(rows):
        if to_skip == 0 and loss > 0 and rng.random() < loss / burst:
            to_skip = burst
        if to_skip > 0:
            to_skip -= 1
            continue
        yield pack_frame(seq, row)


class Receiver:
    """Checks every frame and hands rows to the decoder.

    fill decides what happens when frames went missing:
      "ignore"  nothing: the decoder never learns that time went missing
      "zeros"   one row of zeros per lost frame (the features are normalised, so 0 means average)
      "hold"    the last good row again, once per lost frame
    """

    def __init__(self, fill="zeros"):
        self.fill = fill
        self.last_seq = -1      # the sender starts at 0, so "the frame before the first" is -1
        self.last_row = np.zeros(N_FEATURES, dtype=np.float32)
        self.gaps = 0           # how many times we noticed a hole
        self.lost = 0           # how many frames those holes add up to
        self.damaged = 0        # frames thrown away because the CRC did not match

    def missing_before(self, seq):
        """How many frames went missing between the last good frame and this one? 0 if none.

        This is the gap check `seq - 1 != last_seq`, turned into a count:
        after frame 41, frame 42 means 0 missing and frame 45 means 3 missing.
        """
        return seq - self.last_seq - 1

    def push(self, frame):
        """Take one frame off the wire. Returns the rows to hand the decoder, oldest first."""
        unpacked = unpack_frame(frame)
        if unpacked is None:
            self.damaged += 1   # a damaged frame is a lost frame: the next good one will show the gap
            return []
        seq, row = unpacked
        missing = self.missing_before(seq)
        rows = []
        if missing > 0:
            self.gaps += 1
            self.lost += missing
            if self.fill == "zeros":
                rows = [np.zeros(N_FEATURES, dtype=np.float32)] * missing
            elif self.fill == "hold":
                rows = [self.last_row] * missing
        self.last_seq, self.last_row = seq, row
        return rows + [row]


def receive(frames, fill="zeros"):
    """Run a whole stream of frames through a Receiver. Returns (rows [n, 512], receiver)."""
    receiver = Receiver(fill)
    rows = [row for frame in frames for row in receiver.push(frame)]
    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32), receiver
    return np.stack(rows), receiver


def main():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(6, N_FEATURES)).astype(np.float32)
    sent = [pack_frame(seq, row) for seq, row in enumerate(rows)]
    print(f"one frame is {FRAME_BYTES} bytes; at 50 frames a second that is "
          f"{FRAME_BYTES * 50 * 8 / 1000:.0f} kbit/s")

    damaged = bytearray(sent[2])
    damaged[100] ^= 0b00000100   # flip a single bit in the payload
    print("frame 2 with one bit flipped ->", unpack_frame(bytes(damaged)), "(the CRC caught it)")

    arrived = sent[:3] + sent[4:]   # frame 3 never arrives
    got, receiver = receive(arrived, fill="zeros")
    print(f"sent 6 frames, frame 3 lost: receiver saw {receiver.gaps} gap, {receiver.lost} frame missing, "
          f"handed the decoder {len(got)} rows (the missing one filled with zeros)")


if __name__ == "__main__":
    main()
