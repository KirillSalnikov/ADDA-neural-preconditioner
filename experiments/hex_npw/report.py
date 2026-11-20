import json
import numpy as np
from experiment import RUN, save


def metrics(rows, mode):
    elapsed = np.array([r["nn"]["elapsed"] for r in rows])
    other = np.array([r[mode]["elapsed"] for r in rows])
    return dict(
        total=float(other.sum()),
        ratio=float(other.sum() / elapsed.sum()),
        geomean=float(np.exp(np.log(other / elapsed).mean())),
        neural_faster=int((other > elapsed).sum()),
        iterations=sum(r[mode]["iterations"] for r in rows),
    )


if __name__ == "__main__":
    rows = json.loads((RUN / "test/summary.json").read_text())
    assert len(rows) == 8 and all(
        r[k]["ok"] for r in rows for k in ("plain", "npw", "fixed", "nn")
    )
    result = {
        group: {k: metrics(subset, k) for k in ("plain", "npw", "fixed", "nn")}
        for group, subset in [
            ("all", rows),
            ("m2.05", [r for r in rows if "m2.05" in r["case"]]),
            ("m2.95", [r for r in rows if "m2.95" in r["case"]]),
        ]
    }
    result["largest_field_error"] = max(r["nn"]["field"] for r in rows)
    save(RUN / "test/aggregate.json", result)
    print(json.dumps(result, indent=2))
