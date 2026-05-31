#!/usr/bin/env python3
"""Convert ADDA FFTDIRECT mode=4 preconditioner to mode=5 x-slab float32.

Mode=4 stores 9 component-major complex128 grids:
  component, z, y, x, re/im

Mode=5 stores x-slab-major complex64 grids:
  x, z, y, component, re/im

The mode=5 layout lets MPI ADDA read each rank's local x-slab with one
contiguous read. ADDA converts values back to doublecomplex in memory, so this
only quantizes storage/import bandwidth.
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np


PRECOND_MAGIC = 0x4E49464C
FFTDIRECT_MODE = 4
FFTDIRECT_XSLAB_F32_MODE = 5


def _read_header(path: str) -> tuple[int, int, int, int, int, int, int, int]:
    with open(path, "rb") as f:
        magic, n, nnz, mode, reserved = struct.unpack("<5Q", f.read(40))
        if magic != PRECOND_MAGIC:
            raise ValueError(f"{path}: invalid preconditioner magic 0x{magic:x}")
        if mode != FFTDIRECT_MODE:
            raise ValueError(f"{path}: expected mode=4 FFTDIRECT, got mode={mode}")
        if nnz != 0:
            raise ValueError(f"{path}: expected nnz=0 for FFTDIRECT, got {nnz}")
        gx, gy, gz = struct.unpack("<3Q", f.read(24))
    return magic, n, nnz, mode, reserved, gx, gy, gz


def convert(input_path: str, output_path: str) -> None:
    _magic, n, _nnz, _mode, _reserved, gx, gy, gz = _read_header(input_path)
    grid_n = gx * gy * gz
    data_offset = 8 * (5 + 3)
    expected_bytes = data_offset + 9 * grid_n * 2 * 8
    actual_bytes = os.path.getsize(input_path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{input_path}: expected {expected_bytes} bytes for {gx}x{gy}x{gz}, "
            f"got {actual_bytes}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(input_path, "rb") as f:
        f.seek(data_offset)
        raw = np.fromfile(f, dtype="<f8", count=9 * grid_n * 2)
    raw = raw.reshape(9, gz, gy, gx, 2)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<5Q", PRECOND_MAGIC, n, 0, FFTDIRECT_XSLAB_F32_MODE, 0))
        f.write(struct.pack("<3Q", gx, gy, gz))
        for x in range(gx):
            slab = np.empty((gz, gy, 9, 2), dtype="<f4")
            for comp in range(9):
                slab[:, :, comp, :] = raw[comp, :, :, x, :].astype("<f4", copy=False)
            f.write(slab.reshape(-1).tobytes())

    in_mb = os.path.getsize(input_path) / (1024**2)
    out_mb = os.path.getsize(output_path) / (1024**2)
    print(
        f"Converted FFTDIRECT mode=4 -> mode=5 x-slab f32: "
        f"grid {gx}x{gy}x{gz}, {in_mb:.1f} MB -> {out_mb:.1f} MB, "
        f"{output_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input mode=4 FFTDIRECT .precond")
    parser.add_argument("--output", required=True, help="Output mode=5 x-slab float32 .precond")
    args = parser.parse_args()

    convert(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
