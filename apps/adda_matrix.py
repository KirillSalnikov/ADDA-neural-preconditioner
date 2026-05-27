"""Construct DDA interaction matrices matching ADDA's formulation.

The DDA interaction matrix A satisfies A·P = E_inc, where:
- A_ii = I_3 (unit diagonal after normalization)
- A_ij = -alpha * G(r_i - r_j) for i != j

G is the free-space Green's tensor (symmetric):
G(r) = exp(ikr)/(4*pi*r) * [k^2*(I-rr)/r + (1-ikr)*(3*rr-I)/r^3]

The matrix is complex symmetric: A^T = A (not Hermitian).

Usage:
    python adda_matrix.py --shape sphere --grid 8 --m_re 1.5 --output_dir ./data/ComplexSymmetric/adda/
"""
import os
import argparse
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import torch
try:
    from data import matrix_to_graph_complex
except ImportError:
    try:
        from apps.data import matrix_to_graph_complex
    except ImportError:
        matrix_to_graph_complex = None


def make_sphere_dipoles(grid_size):
    """Generate dipole positions for a sphere inscribed in grid."""
    center = (grid_size - 1) / 2.0
    radius = grid_size / 2.0
    positions = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                dx = ix - center
                dy = iy - center
                dz = iz - center
                if dx**2 + dy**2 + dz**2 <= radius**2:
                    positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_cube_dipoles(grid_size):
    """Generate dipole positions for a cube."""
    positions = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_ellipsoid_dipoles(grid_size, aspect_ratios=(1.0, 0.5, 0.3)):
    """Generate dipole positions for an ellipsoid."""
    center = (grid_size - 1) / 2.0
    positions = []
    ax, ay, az = [grid_size / 2.0 * a for a in aspect_ratios]
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                dx = (ix - center) / ax
                dy = (iy - center) / ay
                dz = (iz - center) / az
                if dx**2 + dy**2 + dz**2 <= 1.0:
                    positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_cylinder_dipoles(grid_size):
    """Generate dipole positions for a cylinder (axis along z, aspect ratio 1:1)."""
    center_xy = (grid_size - 1) / 2.0
    radius = grid_size / 2.0
    positions = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            dx = ix - center_xy
            dy = iy - center_xy
            if dx**2 + dy**2 <= radius**2:
                for iz in range(grid_size):
                    positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_capsule_dipoles(grid_size):
    """Generate dipole positions for a capsule (cylinder + hemispherical caps, axis along z)."""
    center = (grid_size - 1) / 2.0
    radius = grid_size / 4.0  # capsule radius
    half_cyl = grid_size / 4.0  # half-length of cylindrical part
    positions = []
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                dx = ix - center
                dy = iy - center
                dz = iz - center
                r_xy2 = dx**2 + dy**2
                if r_xy2 <= radius**2:
                    # Cylindrical section
                    if abs(dz) <= half_cyl:
                        positions.append([ix, iy, iz])
                    # Top hemisphere
                    elif dz > half_cyl and r_xy2 + (dz - half_cyl)**2 <= radius**2:
                        positions.append([ix, iy, iz])
                    # Bottom hemisphere
                    elif dz < -half_cyl and r_xy2 + (dz + half_cyl)**2 <= radius**2:
                        positions.append([ix, iy, iz])
    return np.array(positions, dtype=float)


def make_hex_prism_dipoles(D_ff, H, d):
    """Hexagonal prism on a cubic lattice.

    Axis along z. Flat-top orientation: flat faces parallel to x-axis.

    Args:
        D_ff: flat-to-flat diameter (physical units)
        H: height along z (physical units)
        d: lattice spacing (physical units)

    Returns:
        positions: (N, 3) integer grid positions
    """
    # Number of grid cells along each dimension
    # Circumradius R = D_ff / sqrt(3)
    R = D_ff / np.sqrt(3.0)
    nx = int(np.ceil(2 * R / d)) + 2
    ny = int(np.ceil(D_ff / d)) + 2
    nz = int(np.ceil(H / d)) + 2

    # Grid coordinates (integer indices)
    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing='ij')

    # Physical coordinates centered at origin
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    cz = (nz - 1) / 2.0
    x = (gx - cx) * d
    y = (gy - cy) * d
    z = (gz - cz) * d

    # Hexagonal cross-section mask (flat-top, 3 inequalities)
    half_D = D_ff / 2.0
    sqrt3_2 = np.sqrt(3.0) / 2.0
    mask_hex = (
        (np.abs(sqrt3_2 * x + 0.5 * y) <= half_D) &
        (np.abs(y) <= half_D) &
        (np.abs(-sqrt3_2 * x + 0.5 * y) <= half_D)
    )

    # Height mask
    mask_z = np.abs(z) <= H / 2.0

    mask = mask_hex & mask_z

    # Extract integer grid positions
    positions = np.column_stack([
        gx[mask].ravel(),
        gy[mask].ravel(),
        gz[mask].ravel(),
    ])

    return positions


def clausius_mossotti_polarizability(m, d, k):
    """Clausius-Mossotti polarizability for a dipole.

    Args:
        m: complex refractive index
        d: interdipole spacing
        k: wavenumber = 2*pi/lambda

    Returns:
        alpha: complex polarizability
    """
    V = d**3  # dipole volume
    eps = m**2
    alpha = 3 * V / (4 * np.pi) * (eps - 1) / (eps + 2)
    return alpha


def ldr_polarizability(m, d, k, S_prop=0.0):
    """Lattice Dispersion Relation (LDR) polarizability with radiative reaction.

    Matches ADDA's implementation exactly (pol3coef + polMplusRR + polM).
    Draine & Goodman (1993), with radiative reaction correction.

    Args:
        m: complex refractive index
        d: interdipole spacing
        k: wavenumber
        S_prop: propagation direction term = sum of products of squared
                propagation direction components. For default z-propagation, S_prop=0.

    Returns:
        alpha: complex polarizability
    """
    eps = m**2
    V = d**3  # dipole volume
    alpha_cm = (3 / (4 * np.pi)) * V * (eps - 1) / (eps + 2)

    # LDR coefficients (matching ADDA's const.h signs)
    B1 = 1.8915316529870796511106114030718259
    B2 = -0.16484691508771947306079362778185226
    B3 = 1.7700004019321371908592738404451742

    kd = k * d
    # M0 = (B1 + (B2 + B3*S_prop)*eps) * kd^2
    M0 = (B1 + (B2 + B3 * S_prop) * eps) * kd**2
    # Radiative reaction correction
    M = M0 + 2j / 3 * kd**3
    # LDR polarizability: alpha = alpha_CM / (1 - (alpha_CM / V) * M)
    alpha = alpha_cm / (1 - (alpha_cm / V) * M)
    return alpha


def green_tensor(r_vec, k):
    """Compute free-space Green's tensor G(r) in DDA convention.

    Uses the Draine & Flatau (1994) convention WITHOUT the 1/(4*pi) factor,
    matching ADDA's InterTerm_poi formulation:
        G(r) = exp(ikr)/r * [k^2*(I-rr) + (1-ikr)/r^2 * (3*rr - I)]

    The 3/(4*pi) in the Clausius-Mossotti polarizability pairs with this
    convention so that alpha*G gives the correct DDA coupling.

    Args:
        r_vec: displacement vector (3,)
        k: wavenumber

    Returns:
        G: 3x3 complex symmetric tensor
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros((3, 3), dtype=complex)

    rhat = r_vec / r
    kr = k * r
    I3 = np.eye(3)
    rr = np.outer(rhat, rhat)

    # exp(ikr)/r * [k^2*(I-rr) + (1-ikr)/r^2 * (3*rr - I)]
    # NO 1/(4*pi) factor — matches ADDA's InterTerm_poi
    phase = np.exp(1j * kr) / r
    G = phase * (
        k**2 * (I3 - rr)
        + (1 - 1j * kr) / r**2 * (3 * rr - I3)
    )
    return G


def build_interaction_matrix(positions, k, m, d=1.0, pol='ldr',
                              threshold=0.0):
    """Build the DDA interaction matrix.

    A_ii = I (unit diagonal)
    A_ij = -alpha * G(r_ij) / d^3 for i != j

    The matrix is normalized so that A P' = E' where P' = P/alpha.

    Args:
        positions: (N, 3) array of dipole positions (in grid units)
        k: wavenumber = 2*pi/lambda
        m: complex refractive index
        d: interdipole spacing (physical units per grid unit)
        pol: polarizability type ('cm' or 'ldr')
        threshold: drop entries with |a_ij| < threshold

    Returns:
        A: complex symmetric matrix (3N x 3N) in sparse CSR format
        n_dipoles: number of dipoles
    """
    N = len(positions)
    n = 3 * N

    # Compute polarizability
    if pol == 'ldr':
        alpha = ldr_polarizability(m, d, k)
    else:
        alpha = clausius_mossotti_polarizability(m, d, k)

    # The coupling constant in ADDA: cc = alpha * k^2 * d
    # But for the normalized matrix (A_ii = I), we need:
    # A_ij = -alpha * G_ij for i != j (in volume-normalized units)

    rows = []
    cols = []
    vals = []

    # Diagonal blocks: I
    for i in range(N):
        for c in range(3):
            rows.append(3 * i + c)
            cols.append(3 * i + c)
            vals.append(1.0 + 0j)

    # Off-diagonal blocks: -alpha * G_ij
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = (positions[j] - positions[i]) * d
            G = green_tensor(r_vec, k)

            # Block value
            block = -alpha * G

            # Store both (i,j) and (j,i) — symmetric
            for ci in range(3):
                for cj in range(3):
                    val = block[ci, cj]
                    if abs(val) > threshold:
                        # (i, j) block
                        rows.append(3 * i + ci)
                        cols.append(3 * j + cj)
                        vals.append(val)
                        # (j, i) block — symmetric
                        rows.append(3 * j + cj)
                        cols.append(3 * i + ci)
                        vals.append(val)

    A = coo_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n, n)
    ).tocsr()

    return A, N


def generate_adda_problems(output_dir, configs):
    """Generate multiple ADDA-like matrices.

    Args:
        output_dir: directory to save files
        configs: list of dicts with keys: shape, grid, m_re, m_im, wavelength
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, cfg in enumerate(configs):
        shape = cfg['shape']
        grid = cfg['grid']
        m = complex(cfg.get('m_re', 1.5), cfg.get('m_im', 0.0))
        wavelength = cfg.get('wavelength', 2 * np.pi)
        k = 2 * np.pi / wavelength
        d = wavelength / (grid * 1.0)  # dipoles per wavelength ~ grid
        threshold = cfg.get('threshold', 1e-10)

        # Generate dipole positions
        if shape == 'sphere':
            positions = make_sphere_dipoles(grid)
        elif shape == 'cube':
            positions = make_cube_dipoles(grid)
        elif shape == 'ellipsoid':
            positions = make_ellipsoid_dipoles(grid,
                            cfg.get('aspect', (1.0, 0.5, 0.3)))
        else:
            raise ValueError(f"Unknown shape: {shape}")

        n_dipoles = len(positions)
        n = 3 * n_dipoles

        print(f"\n[{idx}] shape={shape}, grid={grid}, m={m}, "
              f"N_dip={n_dipoles}, matrix_size={n}")

        # Build interaction matrix
        A, _ = build_interaction_matrix(positions, k, m, d=d,
                                         threshold=threshold)

        print(f"  nnz={A.nnz} ({100*A.nnz/n**2:.1f}%)")

        # Verify properties
        diff_sym = abs(A - A.T).max()
        diag_err = np.max(np.abs(A.diagonal() - 1.0))
        print(f"  symmetry |A-A^T| = {diff_sym:.2e}, "
              f"diag |A_ii-1| = {diag_err:.2e}")

        # Random complex RHS (simulating incident field)
        rng = np.random.RandomState(42 + idx)
        b = rng.randn(n) + 1j * rng.randn(n)
        b = b / np.linalg.norm(b)

        # Save as PyG graph
        if matrix_to_graph_complex is None:
            raise ImportError(
                "apps.data is unavailable; cannot convert matrices to PyG graphs"
            )
        graph = matrix_to_graph_complex(A, b)
        name = f"{n}_{idx}"
        torch.save(graph, os.path.join(output_dir, f"{name}.pt"))

        # Save raw for reference
        A_coo = coo_matrix(A)
        np.savez(
            os.path.join(output_dir, f"{name}_raw.npz"),
            data=A_coo.data, row=A_coo.row, col=A_coo.col,
            shape=A_coo.shape, b=b,
            config=cfg
        )

        print(f"  Saved {name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="./data/ComplexSymmetric/adda/")
    args = parser.parse_args()

    # Generate a range of ADDA-like problems
    configs = []

    # Small spheres (similar to training data size)
    for grid in [4, 5, 6]:
        for m_re in [1.3, 1.5, 2.0]:
            configs.append({
                'shape': 'sphere', 'grid': grid,
                'm_re': m_re, 'm_im': 0.0
            })

    # Medium spheres
    for grid in [8, 10]:
        for m_re in [1.3, 1.5, 2.0]:
            configs.append({
                'shape': 'sphere', 'grid': grid,
                'm_re': m_re, 'm_im': 0.0
            })

    # Absorbing particles (complex refractive index)
    for grid in [6, 8]:
        for m_re, m_im in [(1.5, 0.1), (1.5, 0.5), (0.2, 1.0)]:
            configs.append({
                'shape': 'sphere', 'grid': grid,
                'm_re': m_re, 'm_im': m_im
            })

    # Ellipsoids
    for grid in [6, 8]:
        configs.append({
            'shape': 'ellipsoid', 'grid': grid,
            'm_re': 1.5, 'm_im': 0.0,
            'aspect': (1.0, 0.7, 0.5)
        })

    # Cubes (sharp edges, harder for DDA)
    for grid in [4, 5, 6]:
        configs.append({
            'shape': 'cube', 'grid': grid,
            'm_re': 1.5, 'm_im': 0.0
        })

    # Large sphere for OOD test
    configs.append({
        'shape': 'sphere', 'grid': 12,
        'm_re': 1.5, 'm_im': 0.0
    })

    generate_adda_problems(args.output_dir, configs)
    print(f"\nGenerated {len(configs)} ADDA matrices")
