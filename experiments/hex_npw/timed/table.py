import csv
import json
from gate import RUN, TEST
from experiment import save


def main():
    d = RUN / "test"
    rows = json.loads((d / "summary.json").read_text())
    assert len(rows) == len(TEST)
    modes = ("plain", "npw", "fixed", "gate_fixed", "model")
    with (d / "times.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "h_over_D",
                "g",
                "m",
                "action",
                "method",
                "converged",
                "median_iterations",
                "median_total_s",
                "max_relative_field_error",
                "max_residual",
                "trial_1_s",
                "trial_2_s",
                "trial_3_s",
            ]
        )
        for c, row in zip(TEST, rows):
            for k in modes:
                v = row[k]
                writer.writerow(
                    [
                        *c,
                        row["action"],
                        k,
                        v["ok"],
                        v["iterations"],
                        v["elapsed"],
                        v["field"],
                        v["residual"],
                        *v["samples"],
                    ]
                )
    strata = {}
    for m in sorted({c[2] for c in TEST}):
        group = [r for c, r in zip(TEST, rows) if c[2] == m]
        strata[str(m)] = {
            k: dict(
                total=sum(r[k]["elapsed"] for r in group),
                iterations=sum(r[k]["iterations"] for r in group),
                max_field=max(
                    (r[k]["field"] for r in group if r[k]["field"] is not None),
                    default=None,
                ),
                ok=all(r[k]["ok"] for r in group),
            )
            for k in modes
        }
    save(d / "by_index.json", strata)


if __name__ == "__main__":
    main()
