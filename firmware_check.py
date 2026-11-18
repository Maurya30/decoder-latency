"""Build the decoder for an ARM Cortex-M, read its flash and RAM size, and check what it computes.

    python firmware_check.py

Needs Rust with the thumbv7em-none-eabihf target (rustup target add thumbv7em-none-eabihf).
The run needs QEMU (qemu-system-arm); without it only the sizes are reported.

Flash is the chip's read-only storage: the code and the model's weights live there.
RAM is what changes while it runs: the decoder's working memory and the stack.
Both sizes are read from the linked image in decoder-rs/cortex-m, not estimated.

The run is in an emulator: QEMU's model of a Cortex-M4. It shows that the ARM build
boots and gives exactly the integers Python gives. It is NOT a timing. An emulator's
speed says nothing about a real chip, so no time is measured or reported here.
"""

import datetime
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from data import N_FEATURES
from model import ROOT, RUNS, SmallCNN, shown
from quantize import int_forward, load_int8, quantize_rows

CRATE = ROOT / "decoder-rs" / "cortex-m"
TARGET = "thumbv7em-none-eabihf"
IMAGE = CRATE / "target" / TARGET / "release" / "firmware"
MODEL = RUNS / "cnn_int8.bin"
RESULTS = RUNS / "firmware.json"
STEPS = 200   # the same as STEPS in decoder-rs/cortex-m/src/main.rs
QEMU = ["qemu-system-arm", "-cpu", "cortex-m4", "-machine", "mps2-an386", "-nographic",
        "-semihosting-config", "enable=on,target=native", "-kernel"]


def sections(path):
    """The sections of a 32-bit ELF image that take up space on the chip: {name: (bytes, in_flash, in_ram)}.

    A section is in flash if the file holds its contents, and in RAM if it can be written.
    (.data would be in both: its starting values sit in flash and get copied to RAM at power-on.)
    """
    blob = Path(path).read_bytes()
    assert blob[:5] == b"\x7fELF\x01", "not a 32-bit ELF file"
    table_at, = struct.unpack_from("<I", blob, 0x20)
    entry_size, count, names_index = struct.unpack_from("<3H", blob, 0x2E)
    headers = [struct.unpack_from("<10I", blob, table_at + i * entry_size) for i in range(count)]
    names_at = headers[names_index][4]
    found = {}
    for name_at, kind, flags, _, _, size, *_ in headers:
        name = blob[names_at + name_at: blob.index(b"\0", names_at + name_at)].decode()
        allocated, writable, no_contents = flags & 2, flags & 1, kind == 8
        if allocated and size:
            found[name] = (size, not no_contents, bool(writable))
    return found


def made_up_rows(n_rows, seed=1):
    """The same made-up feature rows as `Noise` in the firmware: values between -2 and 2."""
    values = np.empty(n_rows * N_FEATURES, dtype=np.float32)
    state = seed
    for i in range(len(values)):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        values[i] = ((state >> 8) % 257) / 64.0 - 2.0
    return values.reshape(n_rows, N_FEATURES)


def checksum(scores):
    """The same checksum as `mix` in the firmware: FNV-1a over every score's four bytes."""
    total = 2166136261
    for byte in scores.astype("<i4").tobytes():
        total = ((total ^ byte) * 16777619) & 0xFFFFFFFF
    return total


def main():
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(cargo).exists():
        raise SystemExit("Rust is not installed: see https://rustup.rs")
    subprocess.run([cargo, "build", "--release", "--quiet"], cwd=CRATE, check=True)

    parts = sections(IMAGE)
    flash = sum(size for size, in_flash, _ in parts.values() if in_flash)
    ram = sum(size for size, _, in_ram in parts.values() if in_ram)
    model_bytes = MODEL.stat().st_size
    print(f"{shown(IMAGE)}, built for {TARGET} (ARM Cortex-M4/M7 with a float unit)")
    for name, (size, in_flash, in_ram) in parts.items():
        where = " and ".join(place for place, there in (("flash", in_flash), ("RAM", in_ram)) if there)
        print(f"  {name:<15}{size:>10,} bytes  {where}")
    print(f"  flash {flash:,} bytes: {model_bytes:,} of them are the model file, the rest is code and the vector table")
    print(f"  RAM   {ram:,} bytes of fixed working memory, before the stack")
    result = {"measured": datetime.date.today().isoformat(), "target": TARGET, "model": str(shown(MODEL)),
              "sections": {name: size for name, (size, _, _) in parts.items()},
              "flash_bytes": flash, "model_bytes": model_bytes, "static_ram_bytes": ram}

    if shutil.which(QEMU[0]):
        done = subprocess.run([*QEMU, str(IMAGE)], stdout=subprocess.PIPE, text=True, timeout=600)
        said = dict(line.split() for line in done.stdout.splitlines() if len(line.split()) == 2)
        q = load_int8(MODEL)
        rows = made_up_rows(SmallCNN.window + (STEPS - 1) * SmallCNN.hop)
        scores = int_forward(q, quantize_rows(rows, q["in_inv_scale"]))
        same = (done.returncode == 0 and int(said.get("steps", -1)) == len(scores) == STEPS
                and int(said.get("checksum", -1)) == checksum(scores))
        print(f"\nrun in QEMU ({QEMU[2]}, {QEMU[4]}): {said.get('steps')} output steps on made-up rows")
        print(f"  checksum of every score: firmware {said.get('checksum')}, Python {checksum(scores)}: "
              + ("identical" if same else "DIFFERENT"))
        print(f"  deepest the stack got: {said.get('stack_bytes')} bytes (measured inside the emulated chip)")
        print("  no time was measured: an emulator's speed is not a chip's speed")
        result.update({"emulator": f"qemu-system-arm {QEMU[2]} {QEMU[4]}", "steps": int(said.get("steps", -1)),
                       "checksum_matches_python": same, "stack_bytes": int(said.get("stack_bytes", -1)),
                       "decoder_bytes": int(said.get("decoder_bytes", -1))})
        if not same:
            raise SystemExit("the firmware does not match the Python model")
    else:
        print("\nqemu-system-arm is not installed, so the image was built and measured but not run")

    RESULTS.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {shown(RESULTS)}")


if __name__ == "__main__":
    main()
