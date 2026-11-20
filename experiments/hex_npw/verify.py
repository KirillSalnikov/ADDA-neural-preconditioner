import hashlib
import json
import random
import statistics
import tempfile
from pathlib import Path
import torch
from learn import predict
from experiment import weights_file, HERE, RUN, TEST, name, save, solve


def main():
    binary = HERE / "native/src/seq/adda"
    ck = torch.load(weights_file("best.pt"), weights_only=True)
    fixed = json.loads(weights_file("fit.json").read_text())["fixed"]
    root = RUN / "test"
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        weights=hashlib.sha256(weights_file("best.pt").read_bytes()).hexdigest(),
        binary=hashlib.sha256(binary.read_bytes()).hexdigest(),
        actions=ck["actions"],
        test=TEST,
        fixed=fixed,
        repeat=4,
        eps=5,
        reference_eps=9,
    )
    p = root / "manifest.json"
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(manifest))
    else:
        save(p, manifest)
    rows = []
    for i, c in enumerate(TEST):
        d = root / name(c)
        d.mkdir(exist_ok=True)
        ref = d / "reference"
        if not (d / "ref.json").exists():
            for method in ("bicg", "fgmres"):
                with tempfile.TemporaryDirectory(dir=d) as tmp:
                    o = Path(tmp) / "out"
                    row = solve(
                        c,
                        0,
                        o,
                        eps=9,
                        limit=6000,
                        field=True,
                        solver=method,
                        binary=binary,
                    )
                    if row["ok"]:
                        row["solver"] = method
                        o.rename(ref)
                        save(d / "ref.json", row)
                        break
            else:
                raise RuntimeError(f"No strict reference for {c}")
        modes = {"plain": "plain", "npw": 0, "fixed": fixed, "nn": "model"}
        order = list(modes)
        random.Random(610 + i).shuffle(order)
        action = predict(ck, c)
        summary = {"case": name(c), "action": action, "fixed": fixed}
        for r in range(4):
            for k in order[r:] + order[:r]:
                receipt = d / f"{k}_{r}.json"
                if receipt.exists():
                    continue
                with tempfile.TemporaryDirectory(dir=d) as tmp:
                    row = solve(
                        c,
                        modes[k],
                        Path(tmp) / "out",
                        limit=3000,
                        field=True,
                        ref=ref,
                        binary=binary,
                    )
                if k == "nn":
                    assert row["action"] == action, (row, action)
                save(receipt, row)
        for k in modes:
            series = [json.loads((d / f"{k}_{r}.json").read_text()) for r in range(4)]
            summary[k] = dict(
                ok=all(x["ok"] for x in series),
                iterations=statistics.median(x["iters"] for x in series),
                elapsed=statistics.median(x["elapsed"] for x in series),
                samples=[x["elapsed"] for x in series],
                field=max(
                    (x["field_error"] for x in series if "field_error" in x),
                    default=None,
                ),
            )
        rows.append(summary)
        save(root / "summary.json", rows)
        print(
            name(c),
            action,
            {
                k: (
                    summary[k]["ok"],
                    summary[k]["iterations"],
                    round(summary[k]["elapsed"], 3),
                )
                for k in modes
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
