#!/usr/bin/env python3
"""Convert ADDA ConvSAI mode=3 preconditioner to FFTDIRECT mode=4.

Mode=3 stores a spatial stencil and ADDA rebuilds the frequency-domain Phat
with 9 FFTs during import. Mode=4 stores Phat directly, so ADDA import only
reads the already prepared frequency kernel. This is useful for repeated runs
and orientation averaging, where startup/import overhead should be minimal.
"""

import argparse
import os
import struct

import numpy as np


PRECOND_MAGIC = 0x4E49464C
CONVSAI_MODE = 3
FFTDIRECT_MODE = 4


def read_convsai(path):
    with open(path, "rb") as f:
        header = struct.unpack("<5Q", f.read(40))
        magic, n, n_stencil, mode, reserved = header
        if magic != PRECOND_MAGIC:
            raise ValueError(f"{path}: invalid preconditioner magic 0x{magic:x}")
        if mode != CONVSAI_MODE:
            raise ValueError(f"{path}: expected mode=3 ConvSAI, got mode={mode}")

        stencil = np.frombuffer(f.read(n_stencil * 3 * 4), dtype="<i4").reshape(n_stencil, 3).copy()
        raw = np.frombuffer(f.read(n_stencil * 18 * 8), dtype="<f8")
        if raw.size != n_stencil * 18:
            raise ValueError(f"{path}: truncated ConvSAI kernel")
        kernel = (raw[0::2] + 1j * raw[1::2]).reshape(n_stencil, 3, 3)

    return n, reserved, stencil, kernel


def write_fftdirect(input_path, output_path, gx, gy, gz):
    n, _reserved, stencil, kernel = read_convsai(input_path)
    grid_n = gx * gy * gz
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<5Q", PRECOND_MAGIC, n, 0, FFTDIRECT_MODE, 0))
        f.write(struct.pack("<3Q", gx, gy, gz))

        for a in range(3):
            for b in range(3):
                spatial = np.zeros((gz, gy, gx), dtype=np.complex128)
                spatial[
                    np.mod(stencil[:, 2], gz),
                    np.mod(stencil[:, 1], gy),
                    np.mod(stencil[:, 0], gx),
                ] = kernel[:, a, b]

                phat = np.fft.fftn(spatial)
                flat = phat.reshape(grid_n)
                interleaved = np.empty(2 * grid_n, dtype="<f8")
                interleaved[0::2] = flat.real
                interleaved[1::2] = flat.imag
                f.write(interleaved.tobytes())

    print(
        f"Converted ConvSAI mode=3 -> FFTDIRECT mode=4: "
        f"grid {gx}x{gy}x{gz}, stencil={len(stencil)}, "
        f"size={os.path.getsize(output_path) / (1024 ** 2):.1f} MB -> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input mode=3 ConvSAI .precond")
    parser.add_argument("--output", required=True, help="Output mode=4 FFTDIRECT .precond")
    parser.add_argument("--grid-x", type=int, required=True, help="ADDA FFT gridX, not the particle grid")
    parser.add_argument("--grid-y", type=int, required=True, help="ADDA FFT gridY, not the particle grid")
    parser.add_argument("--grid-z", type=int, required=True, help="ADDA FFT gridZ, not the particle grid")
    args = parser.parse_args()

    write_fftdirect(args.input, args.output, args.grid_x, args.grid_y, args.grid_z)


if __name__ == "__main__":
    main()
