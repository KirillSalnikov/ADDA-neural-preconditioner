"""Train the decision to build the preconditioner on full, paired native runtimes."""

import json
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiment import weights_file, TRAIN, name, save, solve
from learn import predict

RUN = HERE / "run"
PRIMARY = [(h, g, m) for h in (0.65, 1.1) for g in (20, 22) for m in (2.15, 3.0)]
UPPER = [(h, g, 3.2) for h in (0.65, 1.1) for g in (20, 22)]
TEST = PRIMARY + UPPER
BINARY = HERE.parent / "native/src/seq/adda"


def inputs(c):
    h, g, m = c
    # Use the occupied z span, which ADDA also uses at inference.
    zratio = math.floor(g * h + 0.5) / g
    material = m - 2.5
    size = (g - 21) / 3
    return [
        1.0,
        material,
        size,
        zratio - 0.85,
        material * size,
        material * (zratio - 0.85),
    ]


def fit():
    RUN.mkdir(parents=True, exist_ok=True)
    ck = torch.load(weights_file("best.pt"), weights_only=True)
    rows = []
    for c in TRAIN:
        folder = RUN / "train" / name(c)
        folder.mkdir(parents=True, exist_ok=True)
        a = predict(ck, c)
        # Pair evaluation order, switch it on the second repetition.
        for r in range(2):
            for k, act in (
                [("plain", "plain"), ("nn", "model")]
                if r == 0
                else [("nn", "model"), ("plain", "plain")]
            ):
                receipt = folder / f"{r}_{k}.json"
                if not receipt.exists():
                    with tempfile.TemporaryDirectory(dir=folder) as tmp:
                        row = solve(
                            c,
                            act,
                            Path(tmp) / "out",
                            limit=1800,
                            field=True,
                            binary=BINARY,
                        )
                    if act == "model":
                        assert row["action"] == a
                    save(receipt, row)
        vals = {}
        for k in ("plain", "nn"):
            rep = [json.loads((folder / f"{r}_{k}.json").read_text()) for r in range(2)]
            # A non-converged solve is censored; use its elapsed time as a lower bound.
            vals[k] = np.median([t["elapsed"] for t in rep])
            vals[k + "_ok"] = all(t["ok"] for t in rep)
        ratio = np.log(vals["plain"] / vals["nn"])
        if not vals["plain_ok"] and vals["nn_ok"]:
            ratio = max(ratio, 0.7)
        if not vals["nn_ok"] and vals["plain_ok"]:
            ratio = min(ratio, -0.7)
        if not (vals["plain_ok"] or vals["nn_ok"]):
            continue
        rows.append((inputs(c), np.clip(ratio, -2.0, 2.0)))
    x, y = map(np.array, zip(*rows))
    penalty = np.eye(x.shape[1]) * 0.8
    penalty[0, 0] = 0.0
    w = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    save(
        RUN / "gate.json",
        dict(weights=w.tolist(), train=TRAIN, test=TEST, threshold=0.0),
    )
    (HERE.parent / "native/src/hex_time_gate.h").write_text(
        "static const double gate_weight[6]={"
        + ",".join(map(repr, w.tolist()))
        + "};\n"
    )
    print(
        "gate", w, "active", [(c, np.dot(w, inputs(c)) > 0) for c in TEST], flush=True
    )


if __name__ == "__main__":
    fit()
