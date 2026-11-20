"""Single-material cubic voxel geometries in ADDA make_particle.c iteration order."""

import math

import numpy as np


def c_round(value):
    """C's round(): halves are rounded away from zero (Python's round() rounds halves to even)."""
    return int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)


def _shape_constants(shape, params):
    """Return (yx_ratio, zx_ratio, constants dict) as set by ADDA's InitShape for one shape."""
    p = [float(v) for v in params]
    if shape == "sphere":
        if p:
            raise ValueError("sphere takes no parameters")
        return 1.0, 1.0, {}
    if shape == "box":
        if not p:
            return 1.0, 1.0, {"haspY": 0.5, "haspZ": 0.5}
        if len(p) != 2:
            raise ValueError("box takes 0 or 2 parameters (y/x z/x)")
        return p[0], p[1], {"haspY": p[0] / 2, "haspZ": p[1] / 2}
    if shape == "ellipsoid":
        if len(p) != 2:
            raise ValueError("ellipsoid takes 2 parameters (y/x z/x)")
        return p[0], p[1], {"invsqY": 1 / (p[0] * p[0]), "invsqZ": 1 / (p[1] * p[1])}
    if shape == "cylinder":
        if len(p) != 1:
            raise ValueError("cylinder takes 1 parameter (h/d)")
        return 1.0, p[0], {"hdratio": p[0] / 2}
    if shape == "capsule":
        if len(p) != 1:
            raise ValueError("capsule takes 1 parameter (h/d)")
        return 1.0, p[0] + 1, {"hdratio": p[0] / 2}
    if shape == "plate":
        if len(p) != 1 or not 0 < p[0] <= 1:
            raise ValueError("plate takes 1 parameter h/d with 0 < h/d <= 1")
        hdratio = p[0] / 2
        return 1.0, p[0], {"hdratio": hdratio, "ri_2": (0.5 - hdratio) ** 2}
    if shape == "prism":
        if len(p) != 2:
            raise ValueError("prism takes 2 parameters (number of sides, h/Dx)")
        n_sides = int(p[0])
        if n_sides != p[0] or n_sides < 3:
            raise ValueError("prism: number of sides must be an integer >= 3")
        # make_particle.c:1757-1800, side length a = 1.
        prang = math.pi / n_sides
        rc = 0.5 / math.sin(prang)  # circumscribed radius
        ri = rc * math.cos(prang)  # inscribed radius
        if n_sides % 2 == 1:
            dx = rc + ri
            sx = (rc - ri) / 2
            dy = 2 * rc * math.sin((n_sides - 1) * prang / 2)
        else:
            dx = 2 * ri
            sx = 0.0
            dy = 2 * rc if n_sides & 2 else dx
        return (
            dy / dx,
            p[1],
            {
                "prang": prang,
                "xcenter": sx / dx,
                "hdratio": p[1] / 2,
                "rc_2": rc * rc / (dx * dx),
                "ri_2": ri * ri / (dx * dx),
            },
        )
    if shape == "bisphere":
        if len(p) != 1 or p[0] < 0:
            raise ValueError("bisphere takes 1 non-negative parameter (R_cc/d)")
        return 1.0, p[0] + 1, {"hdratio": p[0] / 2}
    raise ValueError(f"unsupported shape: {shape}")


def adda_box(shape, params, grid):
    """ADDA's box (boxX, boxY, boxZ) for -grid GRID -shape SHAPE PARAMS (cubic dipoles, jagged 1)."""
    yx_ratio, zx_ratio, _ = _shape_constants(shape, params)
    box_y = c_round(yx_ratio * grid)
    box_z = c_round(zx_ratio * grid)
    if box_y == 0 or box_z == 0:
        raise ValueError(
            f"{shape} {params} at grid {grid} is too thin (box {grid}x{box_y}x{box_z})"
        )
    return grid, box_y, box_z


def _inside(shape, c, xr, yr, zr):
    """Boolean mask of the voxels inside the particle (ADDA's MakeParticle switch, mat = 0)."""
    if shape == "sphere":
        return xr * xr + yr * yr + zr * zr <= 0.25
    if shape == "box":
        return (
            (np.abs(yr) <= c["haspY"])
            & (np.abs(zr) <= c["haspZ"])
            & np.ones_like(xr, dtype=bool)
        )
    if shape == "ellipsoid":
        return xr * xr + yr * yr * c["invsqY"] + zr * zr * c["invsqZ"] <= 0.25
    if shape == "cylinder":
        return (xr * xr + yr * yr <= 0.25) & (np.abs(zr) <= c["hdratio"])
    if shape == "capsule":
        ro2 = xr * xr + yr * yr
        t = np.abs(zr) - c["hdratio"]
        return (ro2 <= 0.25) & ((t <= 0) | (t * t + ro2 <= 0.25))
    if shape == "plate":
        ro2 = xr * xr + yr * yr
        h = c["hdratio"]
        t = np.sqrt(ro2) - 0.5 + h  # distance from the inner radius ri along ro
        rim = t * t + zr * zr <= h * h
        return (ro2 <= 0.25) & (np.abs(zr) <= h) & ((ro2 <= c["ri_2"]) | rim)
    if shape == "prism":
        xs = xr - c["xcenter"]
        ro2 = xs * xs + yr * yr
        prang = c["prang"]
        candidate = (ro2 <= c["rc_2"]) & (np.abs(zr) <= c["hdratio"])
        # Same expression as ADDA: cos(fmod(|atan2(y, x)| + prang, 2 prang) - prang); np.fmod is C's fmod.
        t = np.cos(np.fmod(np.abs(np.arctan2(yr, xs)) + prang, 2 * prang) - prang)
        return candidate & ((ro2 <= c["ri_2"]) | (t * t * ro2 <= c["ri_2"]))
    if shape == "bisphere":
        ro2 = xr * xr + yr * yr
        t = np.abs(zr) - c["hdratio"]
        return (ro2 <= 0.25) & (t * t + ro2 <= 0.25)
    raise ValueError(f"unsupported shape: {shape}")


def adda_positions(shape, params, grid):
    """Dipole positions of ADDA's particle, as 0-based box indices, in ADDA's order.

    Returns (positions, box): positions is an (N, 3) int32 array of (ix, iy, iz), z outermost and x
    fastest (ADDA's loop order); box is (boxX, boxY, boxZ).
    """
    _, _, constants = _shape_constants(shape, params)
    box = adda_box(shape, params, grid)
    box_x, box_y, box_z = box
    # Index grids with z outermost and x fastest, so a C-order ravel gives ADDA's order.
    kz, jy, ix = np.meshgrid(
        np.arange(box_z), np.arange(box_y), np.arange(box_x), indexing="ij", sparse=True
    )
    # Voxel centres in units of the particle's x size, exactly as make_particle.c computes them.
    xr = (0.5 * (2 * ix + 1 - box_x)) / box_x
    yr = (0.5 * (2 * jy + 1 - box_y)) / box_x
    zr = (0.5 * (2 * kz + 1 - box_z)) / box_x
    mask = _inside(shape, constants, xr, yr, zr)
    positions = np.column_stack(np.nonzero(mask)[::-1]).astype(np.int32)
    if positions.shape[0] == 0:
        raise ValueError(f"all voxels of {shape} {params} at grid {grid} are void")
    return positions, box


def make_shape_positions(shape, grid, ay=1.0, az=1.0):
    if shape in ("hex", "hex_prism"):
        shape, ay = "prism", 6
    if shape == "cube":
        shape = "box"
        ay = az = 1.0
    if shape in ("box", "prism", "ellipsoid"):
        params = (ay, az)
    elif shape == "sphere":
        params = ()
    else:
        params = (az,)
    return adda_positions(shape, params, grid)[0]
