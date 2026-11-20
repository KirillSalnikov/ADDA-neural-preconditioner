import hashlib
import json
import random
import statistics
import tempfile
from pathlib import Path
import numpy as np
import torch
from gate import HERE, RUN, TEST, inputs
from experiment import weights_file, name, save, solve
from learn import predict


def main():
    gate_path = RUN / "gate.json"
    if not gate_path.exists():
        gate_path = HERE.parent / "weights/gate.json"
    weights = np.array(json.loads(gate_path.read_text())["weights"])
    ck = torch.load(weights_file("best.pt"), weights_only=True)
    binary = HERE.parent / "native/src/seq/adda"
    fixed = json.loads(weights_file("fit.json").read_text())["fixed"]
    root = RUN / "test"
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        cases=TEST,
        model=hashlib.sha256(weights_file("best.pt").read_bytes()).hexdigest(),
        gate=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        binary=hashlib.sha256(binary.read_bytes()).hexdigest(),
        fixed=fixed,
        repeats=3,
    )
    path = root / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(manifest))
    else:
        save(path, manifest)
    rows = []
    for i, c in enumerate(TEST):
        folder = root / name(c)
        folder.mkdir(exist_ok=True)
        ref = folder / "reference"
        if not (folder / "ref.json").exists():
            for method in ("bicg", "fgmres"):
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    out = Path(tmp) / "solve"
                    row = solve(
                        c,
                        0,
                        out,
                        eps=9,
                        limit=9000,
                        field=True,
                        solver=method,
                        binary=binary,
                    )
                    if row["ok"]:
                        row["solver"] = method
                        out.rename(ref)
                        save(folder / "ref.json", row)
                        break
            else:
                raise RuntimeError(f"No verified reference for {c}")
        active = float(weights @ inputs(c)) > 0
        a = predict(ck, c) if active else -1
        modes = dict(
            plain="plain",
            npw=0,
            fixed=fixed,
            gate_fixed=fixed if active else "plain",
            model="timed",
        )
        order = list(modes)
        random.Random(i + 529).shuffle(order)
        for r, keys in enumerate((order, list(reversed(order)), order[2:] + order[:2])):
            for key in keys:
                receipt = folder / f"{key}_{r}.json"
                if receipt.exists():
                    continue
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    row = solve(
                        c,
                        modes[key],
                        Path(tmp) / "solve",
                        limit=3500,
                        field=True,
                        ref=ref,
                        binary=binary,
                    )
                if key == "model":
                    assert row["action"] == a, (c, a, row)
                save(receipt, row)
        summary = dict(case=name(c), action=a)
        for key in modes:
            series = [
                json.loads((folder / f"{key}_{r}.json").read_text()) for r in range(3)
            ]
            summary[key] = dict(
                ok=all(v["ok"] for v in series),
                iterations=statistics.median(v["iters"] for v in series),
                elapsed=statistics.median(v["elapsed"] for v in series),
                wall=statistics.median(v["wall"] for v in series),
                samples=[v["elapsed"] for v in series],
                field=max(
                    (v["field_error"] for v in series if "field_error" in v),
                    default=None,
                ),
                residual=max(max(v["residuals"]) for v in series if v["residuals"]),
            )
        rows.append(summary)
        save(root / "summary.json", rows)
        print(
            name(c), a, {k: round(summary[k]["elapsed"], 3) for k in modes}, flush=True
        )
    result = dict(
        accepted=all(row[k]["ok"] for row in rows for k in modes), cases=len(rows)
    )
    for key in modes:
        base = np.array([v[key]["elapsed"] for v in rows])
        neural = np.array([v["model"]["elapsed"] for v in rows])
        result[key] = dict(
            total=base.sum(),
            time_ratio=base.sum() / neural.sum(),
            geomean=np.exp(np.log(base / neural).mean()),
            wins=int((base > neural).sum()),
            iterations=sum(v[key]["iterations"] for v in rows),
        )
    save(root / "aggregate.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
