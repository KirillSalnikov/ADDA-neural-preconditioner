"""Restore the archived training receipts for offline fitting."""

import json

from experiment import ACTIONS, HERE, TRAIN, name, save


def main():
    archive = json.loads((HERE / "benchmarks" / "training.json").read_text())
    paths = {
        "run/train": [f"{index}.json" for index in range(len(ACTIONS))],
        "timed/run/train": [
            f"{r}_{mode}.json" for r in range(2) for mode in ("plain", "nn")
        ],
    }
    missing = []
    for stage, filenames in paths.items():
        for case in TRAIN:
            for filename in filenames:
                key = f"{name(case)}/{filename}"
                receipt = archive[stage][key]
                destination = HERE / stage / key
                if destination.exists():
                    if json.loads(destination.read_text()) != receipt:
                        raise ValueError(
                            f"Different measurement already present in {destination}"
                        )
                else:
                    missing.append((destination, receipt))

    for destination, receipt in missing:
        destination.parent.mkdir(parents=True, exist_ok=True)
        save(destination, receipt)
    print(f"Restored {len(missing)} training receipts")


if __name__ == "__main__":
    main()
