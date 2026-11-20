"""Fit scores of fixed linear preconditioners; choose architecture by held-out-index CV."""

import json
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiment import ACTIONS, HERE, RUN, TRAIN, name, save

sys.path.insert(0, str(HERE.parents[1]))
from apps.adda_geometry import make_shape_positions


def features(case):
    h, g, m = case
    pos = make_shape_positions("prism", g, 6, h)
    pos -= pos.min(0)
    span = pos.max(0) + 1
    width = span.max()
    bins = (4 * pos + 2) // width
    occupancy = np.bincount(bins @ [16, 4, 1], minlength=64) * 64.0 / float(width) ** 3
    return np.array([*occupancy, span[2] / g, (g - 21) / 3, (m - 2.5) / 0.6], dtype="f")


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(64, 5)
        self.hidden = nn.Linear(8, 16)
        self.cost = nn.Linear(16, len(ACTIONS))

    def forward(self, x):
        emb = self.embedding(x[:, :64]).tanh()
        return self.cost(self.hidden(torch.cat((emb, x[:, 64:]), 1)).tanh())


def train_mlp(x, y, work, epochs):
    torch.manual_seed(414)
    model = Policy()
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.02)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (
            F.smooth_l1_loss(model(x), y, reduction="none", beta=0.15).mean(1) * work
        ).mean()
        loss.backward()
        opt.step()
    return model.eval()


def rbf(x, train_x, y, width):
    # Gaussian RBF network with fixed centres on training particles.
    d = x[:, None, :] - train_x[None, :, :]
    distance = (
        (d[:, :, :64] ** 2).mean(2) * 3
        + 4 * d[:, :, 64] ** 2
        + 0.6 * d[:, :, 65] ** 2
        + d[:, :, 66] ** 2
    )
    act = torch.softmax(-distance / width, dim=1)
    return act @ y


def arr(v):
    return "{" + ",".join(arr(i) if isinstance(i, list) else repr(i) for i in v) + "}"


def header(path, arrays):
    out = []
    for k, v in arrays.items():
        a = np.asarray(v)
        dims = "".join(f"[{d}]" for d in a.shape)
        dtype = "double" if k == "hex_actions" else "float"
        out.append(f"static const {dtype} {k}{dims}={arr(a.tolist())};")
    path.write_text("\n".join(out) + "\n")


def fit():
    torch.set_num_threads(1)
    x = torch.tensor(np.stack([features(c) for c in TRAIN]))
    rows = [
        [
            json.loads((RUN / "train" / name(c) / f"{a}.json").read_text())
            for a in range(len(ACTIONS))
        ]
        for c in TRAIN
    ]
    # The primary criterion is convergence of both polarizations. Use a penalty larger than any completed solve.
    cost = torch.tensor(
        [[r["iters"] if r["ok"] else 2 * 800 * 2 for r in rr] for rr in rows],
        dtype=torch.float32,
    )
    if (cost.min(1).values >= 3200).any():
        raise RuntimeError("No converged training action on a particle")
    target = (cost / cost.min(1, keepdim=True).values).log().clamp(max=2.0)
    configs = [("mlp", e) for e in (60, 160, 350)] + [
        ("rbf", w) for w in (0.04, 0.16, 0.5, 1.4)
    ]
    pred = {cfg: torch.zeros(len(x), dtype=torch.long) for cfg in configs}
    # Leave out every row at the same refractive index. No labels from the target indices.
    for m in sorted(set(c[2] for c in TRAIN)):
        valid = np.array([c[2] == m for c in TRAIN])
        xx = x[~valid]
        yy = target[~valid]
        work = cost[~valid].min(1).values.sqrt()
        work = work / work.mean()
        for cfg in configs:
            mode, p = cfg
            model = train_mlp(xx, yy, work, p) if mode == "mlp" else None
            with torch.no_grad():
                scores = (
                    model(x[valid]) if model is not None else rbf(x[valid], xx, yy, p)
                )
            pred[cfg][valid] = scores.argmin(1)

    def criterion(a):
        c = cost.gather(1, a[:, None]).flatten()
        return float(
            0.5 * (c / cost.min(1).values).log().mean()
            + 0.5 * (c.sum() / cost.min(1).values.sum()).log()
        )

    chosen = min(configs, key=lambda cfg: criterion(pred[cfg]))
    fixed = min(range(len(ACTIONS)), key=lambda a: criterion(torch.full((len(x),), a)))
    mode, p = chosen
    arrays = {}
    weights = None
    if mode == "mlp":
        w = cost.min(1).values.sqrt()
        model = train_mlp(x, target, w / w.mean(), p)
        weights = model.state_dict()
        arrays = {k.replace(".", "_"): v.numpy() for k, v in weights.items()}
    else:
        arrays = {"centers": x.numpy(), "costs": target.numpy()}
    arrays["hex_actions"] = [[float(w == "hann"), s, c, ws] for w, s, c, ws in ACTIONS]
    hp = HERE / "native/src/hex_weights.h"
    header(hp, arrays)
    with hp.open("a") as f:
        print(f"#define HEX_RBF {int(mode == 'rbf')}", file=f)
        print(f"#define HEX_WIDTH {float(p)}", file=f)
    path = RUN / "best.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(
        dict(
            mode=mode,
            param=p,
            x=x,
            y=target,
            weights=weights,
            actions=ACTIONS,
            train=TRAIN,
        ),
        tmp,
    )
    tmp.replace(path)
    save(
        RUN / "fit.json",
        dict(
            mode=mode,
            param=p,
            fixed=fixed,
            cv={str(k): criterion(v) for k, v in pred.items()},
            cv_actions=pred[chosen].tolist(),
            training_fixed_score=criterion(torch.full((len(x),), fixed)),
        ),
    )
    print((RUN / "fit.json").read_text())


def predict(ck, case):
    x = torch.tensor(features(case))[None]
    with torch.no_grad():
        if ck["mode"] == "mlp":
            net = Policy()
            net.load_state_dict(ck["weights"])
            scores = net(x)
        else:
            scores = rbf(x, ck["x"], ck["y"], ck["param"])
    return int(scores.argmin())


if __name__ == "__main__":
    fit()
