"""Verify the no-preconditioner comparison against a clean, independently built ADDA."""

import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from gate import RUN, TEST
from experiment import name, save, solve


def main(binary, source_commit):
    fingerprint = hashlib.sha256(binary.read_bytes()).hexdigest()
    provenance = {"commit": source_commit, "binary": fingerprint}
    record = RUN / "test/upstream_inputs.json"
    if record.exists() and json.loads(record.read_text()) != provenance:
        raise RuntimeError(
            "The upstream executable or revision has changed. Use a fresh test directory."
        )
    save(record, provenance)
    summary = []
    for case in TEST:
        folder = RUN / "test" / name(case)
        native = [
            json.loads((folder / f"plain_{r}.json").read_text()) for r in range(3)
        ]
        rows = []
        for r in range(3):
            receipt = folder / f"upstream_{r}.json"
            if not receipt.exists():
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    row = solve(
                        case,
                        "plain",
                        Path(tmp) / "up",
                        limit=3500,
                        field=True,
                        ref=folder / "reference",
                        binary=binary,
                    )
                save(receipt, row)
            rows.append(json.loads(receipt.read_text()))
        summary.append(
            dict(
                case=name(case),
                ok=all(r["ok"] for r in rows),
                elapsed=statistics.median(r["elapsed"] for r in rows),
                iterations=statistics.median(r["iters"] for r in rows),
                field=max(
                    (r["field_error"] for r in rows if "field_error" in r), default=None
                ),
                local_plain=statistics.median(r["elapsed"] for r in native),
                model=statistics.median(
                    json.loads((folder / f"model_{r}.json").read_text())["elapsed"]
                    for r in range(3)
                ),
            )
        )
        print(name(case), summary[-1], flush=True)
    save(
        RUN / "test/upstream.json",
        dict(
            commit=source_commit,
            binary=fingerprint,
            cases=summary,
            all_converged=all(r["ok"] for r in summary),
            total_time_ratio=sum(r["elapsed"] for r in summary)
            / sum(r["model"] for r in summary),
        ),
    )
    print(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Revision used to build the stock executable",
    )
    args = parser.parse_args()
    main(args.binary.resolve(), args.source_commit)
