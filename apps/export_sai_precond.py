"""Export a trained NeuralSAI preconditioner to binary .precond format for ADDA.

Binary format (.precond) — SAI mode:
  Header (40 bytes): magic(u64)=0x4E49464C, n(u64), nnz(u64), mode(u64)=1, reserved(u64)
  Data: row_ptr[n+1](u64), col_idx[nnz](u64), values[nnz](double x 2, re then im)

mode=1 signals left preconditioning: ADDA applies M·A instead of L⁻¹·A·L⁻ᵀ.
M is a general sparse matrix (not lower-triangular).

Usage:
    python apps/export_sai_precond.py --model_dir results/sai_train \
        --shape sphere --grid 8 --m_re 1.5 --kd 1.0 \
        --output /tmp/sphere8_sai.precond
"""
import argparse
import json
import struct
import sys
import os

import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from adda_matrix import (
    make_sphere_dipoles, make_cube_dipoles, make_ellipsoid_dipoles,
    build_interaction_matrix,
)
try:
    from apps.generate_sai_dataset import build_sai_graph
except ImportError:
    build_sai_graph = None
from core.models import NeuralSAI

PRECOND_MAGIC = 0x4E49464C
SAI_MODE = 1  # Left preconditioning flag


def make_sphere_dipoles_adda(grid_size):
    """Generate sphere dipoles in ADDA's enumeration order (z-y-x loops).

    ADDA's make_particle.c uses: for(k=z) for(j=y) for(i=x), so z is the
    outermost loop and x is innermost. This must match for the preconditioner
    matrix rows/columns to align with ADDA's dipole indexing.
    """
    center = (grid_size - 1) / 2.0
    radius = grid_size / 2.0
    positions = []
    for iz in range(grid_size):
        for iy in range(grid_size):
            for ix in range(grid_size):
                dx = ix - center
                dy = iy - center
                dz = iz - center
                if dx**2 + dy**2 + dz**2 <= radius**2:
                    positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_cube_dipoles_adda(grid_size):
    """Generate cube dipoles in ADDA's enumeration order (z-y-x loops)."""
    positions = []
    for iz in range(grid_size):
        for iy in range(grid_size):
            for ix in range(grid_size):
                positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_ellipsoid_dipoles_adda(grid_size, aspect_ratios=(1.0, 0.5, 0.3)):
    """Generate ellipsoid dipoles in ADDA's enumeration order (z-y-x loops)."""
    center = (grid_size - 1) / 2.0
    positions = []
    ax, ay, az = [grid_size / 2.0 * a for a in aspect_ratios]
    for iz in range(grid_size):
        for iy in range(grid_size):
            for ix in range(grid_size):
                dx = ix - center
                dy = iy - center
                dz = iz - center
                if (dx/ax)**2 + (dy/ay)**2 + (dz/az)**2 <= 1.0:
                    positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_box_dipoles_adda(grid_size, y_ratio=1.0, z_ratio=1.0):
    """Generate rectangular box dipoles in ADDA's enumeration order (z-y-x loops).

    ADDA box dimensions: grid_x = grid_size, grid_y = round(grid_size * y_ratio),
    grid_z = round(grid_size * z_ratio).  This matches ADDA's `-shape box <y/x> <z/x>`.
    """
    gx = grid_size
    gy = max(1, round(grid_size * y_ratio))
    gz = max(1, round(grid_size * z_ratio))
    positions = []
    for iz in range(gz):
        for iy in range(gy):
            for ix in range(gx):
                positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def load_model(model_dir, device='cpu'):
    """Load a trained NeuralSAI model from a directory."""
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = NeuralSAI(
        latent_size=config.get("latent_size", 64),
        message_passing_steps=config.get("message_passing_steps", 6),
        activation=config.get("activation", "relu"),
    )

    weights_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, 'final_model.pt')
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, config


def export_sai_precond(M_csr, output_path):
    """Write SAI preconditioner M to binary .precond file.

    Args:
        M_csr: scipy CSR matrix (complex), general sparse (not triangular)
        output_path: output file path
    """
    n = M_csr.shape[0]
    nnz = M_csr.nnz

    with open(output_path, 'wb') as f:
        # Header: magic, n, nnz, mode=1 (SAI/left), reserved
        f.write(struct.pack('<5Q', PRECOND_MAGIC, n, nnz, SAI_MODE, 0))

        # row_ptr as uint64
        row_ptr = M_csr.indptr.astype(np.uint64)
        f.write(row_ptr.tobytes())

        # col_idx as uint64
        col_idx = M_csr.indices.astype(np.uint64)
        f.write(col_idx.tobytes())

        # values as pairs of doubles (re, im)
        vals = M_csr.data
        interleaved = np.empty(2 * nnz, dtype=np.float64)
        interleaved[0::2] = vals.real
        interleaved[1::2] = vals.imag
        f.write(interleaved.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"Exported SAI: n={n}, nnz={nnz}, fill={100*nnz/(n*n):.2f}%, "
          f"file={file_size/1024:.1f} KB -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export NeuralSAI preconditioner to binary format for ADDA'
    )
    parser.add_argument('--model_dir', required=True,
                        help='Directory with trained model')
    parser.add_argument('--shape', default='sphere',
                        choices=['sphere', 'cube', 'ellipsoid', 'box'])
    parser.add_argument('--grid', type=int, required=True)
    parser.add_argument('--m_re', type=float, default=1.5)
    parser.add_argument('--m_im', type=float, default=0.0)
    parser.add_argument('--kd', type=float, default=1.0,
                        help='Size parameter kd = k*d')
    parser.add_argument('--y_ratio', type=float, default=1.0,
                        help='Box y/x aspect ratio (for box shape)')
    parser.add_argument('--z_ratio', type=float, default=1.0,
                        help='Box z/x aspect ratio (for box shape)')
    parser.add_argument('--r_cut', type=float, default=3.0,
                        help='Near-field cutoff for graph edges')
    parser.add_argument('--output', required=True,
                        help='Output .precond file path')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    m = complex(args.m_re, args.m_im)
    d = 1.0
    k = args.kd / d

    # Dipole positions in ADDA's enumeration order (z-y-x loops)
    if args.shape == 'sphere':
        positions = make_sphere_dipoles_adda(args.grid)
    elif args.shape == 'cube':
        positions = make_cube_dipoles_adda(args.grid)
    elif args.shape == 'ellipsoid':
        positions = make_ellipsoid_dipoles_adda(args.grid)
    elif args.shape == 'box':
        positions = make_box_dipoles_adda(args.grid, args.y_ratio, args.z_ratio)
    else:
        raise ValueError(f"Unknown shape: {args.shape}")

    n_dipoles = len(positions)
    n = 3 * n_dipoles
    print(f"Shape: {args.shape}, grid: {args.grid}, N_dip: {n_dipoles}, "
          f"matrix: {n}x{n}")

    # Build graph with extended features
    if build_sai_graph is None:
        raise ImportError(
            "apps.generate_sai_dataset is unavailable; SAI export cannot build the graph"
        )
    data = build_sai_graph(positions, k, m, d=d, r_cut=args.r_cut)

    # Load model
    device = args.device
    if device != 'cpu':
        device = f'cuda:{device}' if not device.startswith('cuda') else device
    model, config = load_model(args.model_dir, device=device)

    # Inference
    with torch.inference_mode():
        data_gpu = data.clone()
        data_gpu.x = data_gpu.x.float()
        data_gpu.edge_attr = data_gpu.edge_attr.float()
        if hasattr(data_gpu, 'global_features') and data_gpu.global_features is not None:
            data_gpu.global_features = data_gpu.global_features.float()
        data_gpu = data_gpu.to(device)

        M_csr_torch, _, _ = model(data_gpu)
        M_dense = M_csr_torch.to('cpu').to_dense().clone()

    # Convert torch sparse CSR to scipy CSR
    M_complex = torch.complex(
        M_dense.real.to(torch.float64),
        M_dense.imag.to(torch.float64)
    ).numpy()
    M_scipy = csr_matrix(M_complex)

    # Drop near-zero entries for efficiency
    M_scipy.eliminate_zeros()
    threshold = 1e-10
    M_scipy.data[np.abs(M_scipy.data) < threshold] = 0
    M_scipy.eliminate_zeros()

    print(f"M: shape={M_scipy.shape}, nnz={M_scipy.nnz}, "
          f"diag range: [{abs(M_scipy.diagonal()).min():.4f}, "
          f"{abs(M_scipy.diagonal()).max():.4f}]")

    export_sai_precond(M_scipy, args.output)


if __name__ == '__main__':
    main()
