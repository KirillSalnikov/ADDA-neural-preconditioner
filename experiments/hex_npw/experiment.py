"""Dense material sweep, held-out index interpolation on two hexagonal prisms."""

import concurrent.futures
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
ADDA = HERE / "native/src/seq/adda"


def weights_file(filename):
    trained = RUN / filename
    return trained if trained.is_file() else HERE / "weights" / filename


# Window name, material damping, scale of coupling, correction to the Hann periodization.
ACTIONS = [
    ("hann", 0.05, 1.0, 0.0),
    ("hann", 0.05, 1.0, -0.8),
    ("hann", 0.05, 0.96, 0.0),
    ("hann", 0.05, 1.035, 0.0),
    ("hann", 0.015, 1.0, 0.0),
    ("hann", 0.15, 1.0, 0.0),
    ("tri", 0.05, 1.0, 0.0),
    ("tri", 0.20, 1.0, 0.0),
    ("hann", 0.15, 1.0, -0.8),
]
TRAIN = list(itertools.product((0.65, 1.1), (18, 24), (1.75, 2.3, 2.8, 3.1, 3.25)))
TEST = list(itertools.product((0.65, 1.1), (18, 24), (2.05, 2.95)))


def name(c):
    h, g, m = c
    return f"h{h:g}_g{g}_m{m:g}"


def save(p, x):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def solve(
    c, a, d, eps=5, limit=800, field=False, ref=None, solver="bicgstab", binary=ADDA
):
    h, g, m = c
    env = os.environ.copy()
    for k in (
        "ADDA_NPW_TIME_GATE",
        "ADDA_NPW_HEX_AUTO",
        "ADDA_NPW_RESIDUAL_AUTO",
        "ADDA_NPW_IDENTITY_SHIFT",
        "ADDA_NPW_BOX_SCALE",
        "ADDA_NPW_WINDOW_SHAPE",
    ):
        env.pop(k, None)
    cmd = [
        str(binary),
        "-grid",
        str(g),
        "-shape",
        "prism",
        "6",
        str(h),
        "-m",
        str(m),
        "0",
        "-dpl",
        "30",
        "-int",
        "fcd",
        "-pol",
        "fcd",
        "-iter",
        solver,
        "-eps",
        str(eps),
        "-recalc_resid",
        "-maxiter",
        str(limit),
        "-dir",
        str(d),
    ]
    if a != "plain":
        if a in ("model", "timed"):
            env["ADDA_NPW_HEX_AUTO"] = "1"
        if a == "timed":
            env["ADDA_NPW_TIME_GATE"] = "1"
        window, damping, scale, shape = ACTIONS[0 if a in ("model", "timed") else a]
        cmd += ["-precond_npw", str(damping), window, "-precond_npw_match"]
        env.update(ADDA_NPW_BOX_SCALE=str(scale), ADDA_NPW_WINDOW_SHAPE=str(shape))
    if field:
        cmd.append("-store_int_field")
    start = time.perf_counter()
    result = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    elapsed = time.perf_counter() - start
    log = (d / "log").read_text() if (d / "log").is_file() else result.stdout
    tails = list(
        map(float, re.findall(r"Final \(recalculated\) residual norm:\s*(\S+)", log))
    )
    count = re.search(r"Total number of iterations:\s*(\d+)", log)
    wall = re.search(r"Total wall time:\s*([\d.]+)", log)
    ok = (
        result.returncode == 0
        and count is not None
        and wall is not None
        and len(tails) == 2
        and all(math.isfinite(t) and 0 <= t <= 10**-eps for t in tails)
    )
    if not count and "converg" not in log.lower() and result.returncode:
        raise RuntimeError(log[-800:])
    row = dict(
        ok=bool(ok),
        iters=int(count[1]) if count else None,
        elapsed=elapsed,
        wall=float(wall[1]) if wall else None,
        residuals=tails,
    )
    if a in ("model", "timed"):
        found = re.search(r"Hex NPW action (-?\d+)", result.stdout)
        if not found:
            raise RuntimeError(result.stdout[:1000])
        row["action"] = int(found[1])
    if ref and ok:
        err = []
        for pol in ("X", "Y"):
            a1 = np.loadtxt(d / f"IntField-{pol}", skiprows=1)
            b1 = np.loadtxt(ref / f"IntField-{pol}", skiprows=1)
            assert np.array_equal(a1[:, :3], b1[:, :3])
            err.append(
                float(
                    np.linalg.norm(a1[:, 4:10] - b1[:, 4:10])
                    / np.linalg.norm(b1[:, 4:10])
                )
            )
        row["field_error"] = max(err)
        row["ok"] = row["field_error"] <= 0.002
    return row


def collect(pair):
    c, a = pair
    p = RUN / "train" / name(c)
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{a}.json"
    if f.exists():
        return
    with tempfile.TemporaryDirectory(dir=p) as tmp:
        save(f, solve(c, a, Path(tmp) / "out"))


if __name__ == "__main__":
    RUN.mkdir(exist_ok=True)
    spec = dict(
        train=TRAIN,
        test=TEST,
        actions=ACTIONS,
        criterion="held-out refractive indices; both polarizations; real residual and reference field",
    )
    plan = RUN / "plan.json"
    if plan.exists():
        assert json.loads(plan.read_text()) == json.loads(json.dumps(spec))
    else:
        save(plan, spec)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(collect, itertools.product(TRAIN, range(len(ACTIONS)))))
