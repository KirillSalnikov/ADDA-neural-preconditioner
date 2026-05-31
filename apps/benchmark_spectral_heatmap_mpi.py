#!/usr/bin/env python3
"""Benchmark Spectral preconditioner speedup with MPI ADDA.

For every (grid, m_re) case this script runs:
  1. baseline MPI ADDA without a preconditioner;
  2. export_spectral_precond.py for the same case;
  3. MPI ADDA with the exported .precond file.

Results are written incrementally to CSV/JSONL and plotted as heatmaps, so a
long run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = ROOT / "apps" / "export_spectral_precond.py"

ITER_RE = re.compile(r"Total number of iterations:\s+(\d+)")
WALL_RE = re.compile(r"Total wall time:\s+([0-9.]+)")
OCC_RE = re.compile(r"Total number of occupied dipoles:\s+(\d+)")
RES_RE = re.compile(r"RE_(\d+)\s*=\s*([0-9.DdEe+-]+)")
STENCIL_RE = re.compile(r"\bstencil=(\d+)\b")
EXPORT_N_RE = re.compile(r"\bN=(\d+)\b")


CSV_FIELDS = [
    "grid",
    "m_re",
    "m_im",
    "shape",
    "ay",
    "az",
    "dpl",
    "kd",
    "eps_exp",
    "target_residual",
    "np",
    "maxiter",
    "baseline_iter",
    "precond_iter",
    "fgmres_restart",
    "status",
    "baseline_status",
    "baseline_rc",
    "baseline_converged",
    "baseline_iters",
    "baseline_reached_iter",
    "baseline_last_iter",
    "baseline_final_residual",
    "baseline_min_residual",
    "baseline_wall_s",
    "baseline_elapsed_s",
    "baseline_n_dipoles",
    "export_status",
    "export_rc",
    "export_elapsed_s",
    "export_stencil",
    "export_n_dipoles",
    "precond_file_mb",
    "precond_status",
    "precond_rc",
    "precond_converged",
    "precond_iters",
    "precond_reached_iter",
    "precond_last_iter",
    "precond_final_residual",
    "precond_min_residual",
    "precond_wall_s",
    "precond_elapsed_s",
    "precond_n_dipoles",
    "iter_speedup",
    "adda_wall_speedup",
    "elapsed_speedup",
    "total_elapsed_speedup",
    "case_dir",
    "precond_file",
]


def _parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(int(part))
    return items


def _parse_float_list(value: str) -> list[float]:
    items: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(float(part))
    return items


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _fmt_num(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _run(cmd: list[str], timeout: float | None, env: dict[str, str]) -> tuple[int, float, str, str]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        output = (proc.stdout or "") + (proc.stderr or "")
        status = "ok" if proc.returncode == 0 else f"rc_{proc.returncode}"
        return proc.returncode, elapsed, output, status
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return -124, elapsed, stdout + stderr + "\nTIMEOUT\n", "timeout"


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    fftw_path = os.path.expanduser(args.fftw_lib_path)
    if fftw_path:
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(os.pathsep) if p]
        if fftw_path not in parts:
            env["LD_LIBRARY_PATH"] = fftw_path + (os.pathsep + current if current else "")
    return env


def _shape_name(shape: str) -> str:
    if shape in {"hex", "hex_prism"}:
        return "prism"
    return shape


def _shape_cli(args: argparse.Namespace) -> list[str]:
    shape = _shape_name(args.shape)
    values = ["-shape", shape]
    if shape == "prism":
        values += [str(args.ay), str(args.az)]
    elif shape in {"cylinder", "plate", "capsule"}:
        values += [str(args.az)]
    elif shape in {"ellipsoid", "box"}:
        values += [str(args.ay), str(args.az)]
    return values


def _append_adda_log(work_dir: Path, output: str) -> str:
    log_path = work_dir / "log"
    if not log_path.exists():
        return output
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return output
    if log_text and log_text not in output:
        return output + "\n" + log_text
    return output


def _parse_residuals(text: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for idx, value in RES_RE.findall(text):
        try:
            out.append((int(idx), float(value.replace("D", "E").replace("d", "e"))))
        except ValueError:
            continue
    return out


def _parse_adda(text: str, eps_exp: int) -> dict[str, Any]:
    residuals = _parse_residuals(text)
    target = 10.0 ** (-eps_exp)
    reached_iter = None
    for idx, value in residuals:
        if value <= target:
            reached_iter = idx
            break

    total_iter_match = ITER_RE.search(text)
    wall_match = WALL_RE.search(text)
    occ_match = OCC_RE.search(text)

    total_iters = int(total_iter_match.group(1)) if total_iter_match else None
    last_iter = residuals[-1][0] if residuals else None
    final_residual = residuals[-1][1] if residuals else None
    min_residual = min((value for _, value in residuals), default=None)
    wall_s = float(wall_match.group(1)) if wall_match else None
    n_dipoles = int(occ_match.group(1)) if occ_match else None

    converged = (
        final_residual is not None
        and final_residual <= target
        and "Iterations haven't converged" not in text
    )

    return {
        "converged": converged,
        "iters": total_iters if total_iters is not None else reached_iter if reached_iter is not None else last_iter,
        "reached_iter": reached_iter,
        "last_iter": last_iter,
        "final_residual": final_residual,
        "min_residual": min_residual,
        "wall_s": wall_s,
        "n_dipoles": n_dipoles,
    }


def _run_adda(
    args: argparse.Namespace,
    grid: int,
    m_re: float,
    work_dir: Path,
    precond_file: Path | None,
    label: str,
) -> dict[str, Any]:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    env = _env(args)
    solver = args.precond_iter if precond_file is not None else "bicgstab"
    mpirun_exports = ["-x", f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}"]
    if precond_file is not None and solver == "fgmres":
        env["ADDA_FGMRES_RESTART"] = str(args.fgmres_restart)
        mpirun_exports += ["-x", f"ADDA_FGMRES_RESTART={args.fgmres_restart}"]

    adda_bin = str((ROOT / args.adda_bin).resolve()) if not os.path.isabs(args.adda_bin) else args.adda_bin
    cmd = [
        "mpirun",
        "--oversubscribe",
        "-np",
        str(args.np),
        *mpirun_exports,
        adda_bin,
        "-grid",
        str(grid),
        "-m",
        str(m_re),
        str(args.m_im),
        "-dpl",
        str(args.dpl),
        "-iter",
        solver,
        "-eps",
        str(args.eps_exp),
        "-maxiter",
        str(args.maxiter),
        "-dir",
        str(work_dir),
    ] + _shape_cli(args)
    if precond_file is not None:
        cmd += ["-precond", str(precond_file)]

    rc, elapsed, output, run_status = _run(cmd, args.adda_timeout, env)
    output = _append_adda_log(work_dir, output)
    (work_dir / f"{label}.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (work_dir / f"{label}.output.txt").write_text(output, encoding="utf-8", errors="replace")

    parsed = _parse_adda(output, args.eps_exp)
    parsed.update({"rc": rc, "elapsed_s": elapsed, "status": run_status, "solver": solver})
    if rc == 0 and not parsed["converged"]:
        parsed["status"] = "not_converged"
    return parsed


def _export_precond(
    args: argparse.Namespace,
    grid: int,
    m_re: float,
    case_dir: Path,
    precond_file: Path,
) -> dict[str, Any]:
    env = _env(args)
    if precond_file.exists():
        precond_file.unlink()

    shape = _shape_name(args.shape)
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--checkpoint",
        str(args.checkpoint),
        "--grid",
        str(grid),
        "--m_re",
        str(m_re),
        "--m_im",
        str(args.m_im),
        "--kd",
        str(args.kd),
        "--shape",
        shape,
        "--ay",
        str(args.ay),
        "--az",
        str(args.az),
        "--threshold-rel",
        str(args.threshold_rel),
        "--blend-identity",
        str(args.blend_identity),
        "--symmetry",
        args.symmetry,
        "--output",
        str(precond_file),
    ]
    if args.max_radius is not None:
        cmd += ["--max-radius", str(args.max_radius)]
    if args.normalize_inputs:
        cmd.append("--normalize-inputs")
    if args.spectral_correction_scale is not None:
        cmd += ["--spectral-correction-scale", str(args.spectral_correction_scale)]

    rc, elapsed, output, run_status = _run(cmd, args.export_timeout, env)
    (case_dir / "export.command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    (case_dir / "export.output.txt").write_text(output, encoding="utf-8", errors="replace")

    stencil = None
    n_dipoles = None
    if match := STENCIL_RE.search(output):
        stencil = int(match.group(1))
    if match := EXPORT_N_RE.search(output):
        n_dipoles = int(match.group(1))
    file_mb = precond_file.stat().st_size / (1024 * 1024) if precond_file.exists() else None
    status = "ok" if rc == 0 and precond_file.exists() and precond_file.stat().st_size > 0 else run_status
    return {
        "rc": rc,
        "elapsed_s": elapsed,
        "status": status,
        "stencil": stencil,
        "n_dipoles": n_dipoles,
        "file_mb": file_mb,
    }


def _speedup(numer: Any, denom: Any) -> float | None:
    if not _finite(numer) or not _finite(denom) or float(denom) <= 0.0:
        return None
    return float(numer) / float(denom)


def _row_key(row: dict[str, Any]) -> tuple[int, float]:
    return int(float(row["grid"])), float(row["m_re"])


def _load_rows(csv_path: Path) -> dict[tuple[int, float], dict[str, Any]]:
    if not csv_path.exists():
        return {}
    rows: dict[tuple[int, float], dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("grid") and row.get("m_re"):
                rows[_row_key(row)] = dict(row)
    return rows


def _write_rows(rows_by_key: dict[tuple[int, float], dict[str, Any]], out_dir: Path) -> None:
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    csv_path = out_dir / "results.csv"
    jsonl_path = out_dir / "results.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _render_heatmaps(rows_by_key: dict[tuple[int, float], dict[str, Any]], out_dir: Path) -> None:
    rows = list(rows_by_key.values())
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    grids = sorted({int(float(row["grid"])) for row in rows})
    m_values = sorted({float(row["m_re"]) for row in rows})
    metrics = [
        ("iter_speedup", "Iteration speedup"),
        ("adda_wall_speedup", "ADDA wall-time speedup"),
        ("total_elapsed_speedup", "Wall-time speedup incl. export"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.8), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, metrics):
        data = np.full((len(m_values), len(grids)), np.nan, dtype=float)
        labels = [["" for _ in grids] for _ in m_values]
        for row in rows:
            gi = grids.index(int(float(row["grid"])))
            mi = m_values.index(float(row["m_re"]))
            value = _float_or_none(str(row.get(metric, "")))
            if value is not None and math.isfinite(value):
                data[mi, gi] = value
                labels[mi][gi] = f"{value:.2f}x"
            else:
                status = str(row.get("status", ""))
                labels[mi][gi] = "..." if not status else status.replace("_", "\n")

        positive = data[np.isfinite(data) & (data > 0.0)]
        vmax = max(2.0, float(np.nanmax(positive))) if positive.size else 2.0
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#d8d8d8")
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("grid")
        ax.set_xticks(range(len(grids)))
        ax.set_xticklabels([str(g) for g in grids])
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f"{m:g}" for m in m_values])
        if ax is axes[0]:
            ax.set_ylabel("Re(m)")
        fig.colorbar(im, ax=ax, shrink=0.82)

        for mi in range(len(m_values)):
            for gi in range(len(grids)):
                label = labels[mi][gi]
                if not label:
                    continue
                value = data[mi, gi]
                color = "black" if np.isfinite(value) and value >= 1.4 else "white"
                if not np.isfinite(value):
                    color = "#333333"
                ax.text(gi, mi, label, ha="center", va="center", fontsize=8, color=color)

    done = sum(1 for row in rows if row.get("status") == "ok")
    fig.suptitle(f"Spectral preconditioner MPI ADDA, eps=1e-3 ({done}/{len(rows)} ok)")
    fig.savefig(out_dir / "heatmap_speedup.png", dpi=160)
    plt.close(fig)

    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        data = np.full((len(m_values), len(grids)), np.nan, dtype=float)
        for row in rows:
            gi = grids.index(int(float(row["grid"])))
            mi = m_values.index(float(row["m_re"]))
            value = _float_or_none(str(row.get(metric, "")))
            if value is not None and math.isfinite(value):
                data[mi, gi] = value
        positive = data[np.isfinite(data) & (data > 0.0)]
        vmax = max(2.0, float(np.nanmax(positive))) if positive.size else 2.0
        norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#d8d8d8")
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("grid")
        ax.set_ylabel("Re(m)")
        ax.set_xticks(range(len(grids)))
        ax.set_xticklabels([str(g) for g in grids])
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f"{m:g}" for m in m_values])
        fig.colorbar(im, ax=ax, shrink=0.86)
        for mi in range(len(m_values)):
            for gi in range(len(grids)):
                value = data[mi, gi]
                label = f"{value:.2f}x" if np.isfinite(value) else ""
                if label:
                    color = "black" if value >= 1.4 else "white"
                    ax.text(gi, mi, label, ha="center", va="center", fontsize=9, color=color)
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=160)
        plt.close(fig)

    def fmt_seconds(value: float) -> str:
        if value >= 3600.0:
            return f"{value / 3600.0:.1f}h"
        if value >= 60.0:
            return f"{value / 60.0:.1f}m"
        return f"{value:.1f}s"

    raw_metrics = [
        ("baseline_wall_s", "Baseline ADDA wall time"),
        ("precond_wall_s", "Preconditioned ADDA wall time"),
        ("export_elapsed_s", "Preconditioner export elapsed time"),
    ]
    for metric, title in raw_metrics:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        data = np.full((len(m_values), len(grids)), np.nan, dtype=float)
        for row in rows:
            gi = grids.index(int(float(row["grid"])))
            mi = m_values.index(float(row["m_re"]))
            value = _float_or_none(str(row.get(metric, "")))
            if value is not None and math.isfinite(value) and value > 0.0:
                data[mi, gi] = value
        positive = data[np.isfinite(data) & (data > 0.0)]
        if positive.size:
            norm = LogNorm(vmin=max(float(np.nanmin(positive)), 1e-6), vmax=float(np.nanmax(positive)))
        else:
            norm = None
        cmap = plt.get_cmap("viridis_r").copy()
        cmap.set_bad("#d8d8d8")
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("grid")
        ax.set_ylabel("Re(m)")
        ax.set_xticks(range(len(grids)))
        ax.set_xticklabels([str(g) for g in grids])
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f"{m:g}" for m in m_values])
        fig.colorbar(im, ax=ax, shrink=0.86, label="seconds")
        for mi in range(len(m_values)):
            for gi in range(len(grids)):
                value = data[mi, gi]
                if np.isfinite(value):
                    ax.text(gi, mi, fmt_seconds(float(value)), ha="center", va="center", fontsize=9, color="white")
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=160)
        plt.close(fig)


def _summarize(rows_by_key: dict[tuple[int, float], dict[str, Any]], out_dir: Path) -> None:
    rows = list(rows_by_key.values())
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    def values(name: str) -> list[float]:
        out = []
        for row in ok_rows:
            value = _float_or_none(str(row.get(name, "")))
            if value is not None and math.isfinite(value):
                out.append(value)
        return out

    lines = [
        "Spectral MPI ADDA heatmap summary",
        f"cases={len(rows)} ok={len(ok_rows)}",
    ]
    precond_iters = sorted({str(row.get("precond_iter", "")) for row in rows if row.get("precond_iter")})
    restarts = sorted({str(row.get("fgmres_restart", "")) for row in rows if row.get("fgmres_restart")})
    if precond_iters:
        lines.append(f"precond_iter={','.join(precond_iters)}")
    if restarts:
        lines.append(f"fgmres_restart={','.join(restarts)}")
    for metric in ("iter_speedup", "adda_wall_speedup", "total_elapsed_speedup"):
        vals = values(metric)
        if not vals:
            lines.append(f"{metric}: no valid values")
            continue
        lines.append(
            f"{metric}: min={min(vals):.3f}x mean={sum(vals) / len(vals):.3f}x max={max(vals):.3f}x"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case(args: argparse.Namespace, grid: int, m_re: float, out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / f"g{grid}_m{_fmt_num(m_re)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    precond_file = case_dir / "spectral.precond"

    print(f"\n=== grid={grid} m_re={m_re:g} eps=1e-{args.eps_exp} ===", flush=True)
    baseline = _run_adda(args, grid, m_re, case_dir / "baseline", None, "baseline")
    print(
        "baseline "
        f"status={baseline['status']} iters={baseline['iters']} "
        f"res={baseline['final_residual']} wall={baseline['wall_s']} elapsed={baseline['elapsed_s']:.1f}s",
        flush=True,
    )

    export = _export_precond(args, grid, m_re, case_dir, precond_file)
    print(
        "export "
        f"status={export['status']} stencil={export['stencil']} "
        f"file_mb={export['file_mb']} elapsed={export['elapsed_s']:.1f}s",
        flush=True,
    )

    precond: dict[str, Any]
    if export["status"] == "ok":
        precond = _run_adda(args, grid, m_re, case_dir / "precond", precond_file, "precond")
    else:
        precond = {
            "rc": None,
            "elapsed_s": None,
            "status": "export_failed",
            "converged": False,
            "iters": None,
            "reached_iter": None,
            "last_iter": None,
            "final_residual": None,
            "min_residual": None,
            "wall_s": None,
            "n_dipoles": None,
            "solver": args.precond_iter,
        }
    print(
        "precond "
        f"status={precond['status']} iters={precond['iters']} "
        f"res={precond['final_residual']} wall={precond['wall_s']} elapsed={precond['elapsed_s']}",
        flush=True,
    )

    status = "ok" if baseline["converged"] and precond["converged"] else "not_converged"
    if baseline["status"] == "timeout":
        status = "baseline_timeout"
    elif export["status"] != "ok":
        status = "export_failed"
    elif precond["status"] == "timeout":
        status = "precond_timeout"
    elif baseline["rc"] not in (0, None):
        status = "baseline_failed"
    elif precond["rc"] not in (0, None):
        status = "precond_failed"

    row = {
        "grid": grid,
        "m_re": m_re,
        "m_im": args.m_im,
        "shape": _shape_name(args.shape),
        "ay": args.ay,
        "az": args.az,
        "dpl": args.dpl,
        "kd": args.kd,
        "eps_exp": args.eps_exp,
        "target_residual": 10.0 ** (-args.eps_exp),
        "np": args.np,
        "maxiter": args.maxiter,
        "baseline_iter": baseline.get("solver", "bicgstab"),
        "precond_iter": precond.get("solver", args.precond_iter),
        "fgmres_restart": args.fgmres_restart if args.precond_iter == "fgmres" else None,
        "status": status,
        "baseline_status": baseline["status"],
        "baseline_rc": baseline["rc"],
        "baseline_converged": baseline["converged"],
        "baseline_iters": baseline["iters"],
        "baseline_reached_iter": baseline["reached_iter"],
        "baseline_last_iter": baseline["last_iter"],
        "baseline_final_residual": baseline["final_residual"],
        "baseline_min_residual": baseline["min_residual"],
        "baseline_wall_s": baseline["wall_s"],
        "baseline_elapsed_s": baseline["elapsed_s"],
        "baseline_n_dipoles": baseline["n_dipoles"],
        "export_status": export["status"],
        "export_rc": export["rc"],
        "export_elapsed_s": export["elapsed_s"],
        "export_stencil": export["stencil"],
        "export_n_dipoles": export["n_dipoles"],
        "precond_file_mb": export["file_mb"],
        "precond_status": precond["status"],
        "precond_rc": precond["rc"],
        "precond_converged": precond["converged"],
        "precond_iters": precond["iters"],
        "precond_reached_iter": precond["reached_iter"],
        "precond_last_iter": precond["last_iter"],
        "precond_final_residual": precond["final_residual"],
        "precond_min_residual": precond["min_residual"],
        "precond_wall_s": precond["wall_s"],
        "precond_elapsed_s": precond["elapsed_s"],
        "precond_n_dipoles": precond["n_dipoles"],
        "iter_speedup": _speedup(baseline["iters"], precond["iters"]) if status == "ok" else None,
        "adda_wall_speedup": _speedup(baseline["wall_s"], precond["wall_s"]) if status == "ok" else None,
        "elapsed_speedup": _speedup(baseline["elapsed_s"], precond["elapsed_s"]) if status == "ok" else None,
        "total_elapsed_speedup": _speedup(
            baseline["elapsed_s"],
            (export["elapsed_s"] or 0.0) + (precond["elapsed_s"] or 0.0),
        )
        if status == "ok"
        else None,
        "case_dir": str(case_dir),
        "precond_file": str(precond_file),
    }
    print(
        "result "
        f"status={status} iter_speedup={row['iter_speedup']} "
        f"wall_speedup={row['adda_wall_speedup']} total_speedup={row['total_elapsed_speedup']}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt")
    parser.add_argument("--output-dir", "--out-dir", default=None)
    parser.add_argument("--grids", default="32,48,64,80,96")
    parser.add_argument("--m-re-values", "--m_re_values", default="1.5,2.0,2.5,3.0,3.5")
    parser.add_argument("--m-im", "--m_im", dest="m_im", type=float, default=0.0)
    parser.add_argument("--shape", default="prism")
    parser.add_argument("--ay", type=float, default=6.0)
    parser.add_argument("--az", type=float, default=1.0)
    parser.add_argument("--dpl", type=float, default=15.0)
    parser.add_argument("--kd", type=float, default=0.41887902047863906)
    parser.add_argument("--eps-exp", "--eps", dest="eps_exp", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--np", type=int, default=16)
    parser.add_argument("--adda-bin", default="adda/src/mpi/adda_mpi")
    parser.add_argument("--fftw-lib-path", default="~/.local/lib")
    parser.add_argument("--precond-iter", default="bicgstab", choices=["bicgstab", "fgmres"])
    parser.add_argument("--fgmres-restart", type=int, default=100)
    parser.add_argument("--threshold-rel", type=float, default=1e-6)
    parser.add_argument("--max-radius", type=int, default=40)
    parser.add_argument("--blend-identity", type=float, default=1.0)
    parser.add_argument("--symmetry", default="z180_zflip", choices=["none", "z180", "zflip", "z180_zflip", "d2"])
    parser.add_argument("--spectral-correction-scale", type=float, default=None)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--export-timeout", type=float, default=900.0)
    parser.add_argument("--adda-timeout", type=float, default=1800.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"runs/HEATMAP_MPI_E3_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    args.checkpoint = str(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    adda_path = Path(args.adda_bin)
    if not adda_path.is_absolute():
        adda_path = ROOT / adda_path
    if not adda_path.exists():
        raise FileNotFoundError(f"MPI ADDA binary not found: {adda_path}")

    grids = _parse_int_list(args.grids)
    m_values = _parse_float_list(args.m_re_values)
    if not grids or not m_values:
        raise ValueError("--grids and --m-re-values must be non-empty")

    rows_by_key = _load_rows(out_dir / "results.csv")
    if args.plot_only:
        _render_heatmaps(rows_by_key, out_dir)
        _summarize(rows_by_key, out_dir)
        return 0

    config = {
        "checkpoint": args.checkpoint,
        "adda_bin": str(adda_path),
        "grids": grids,
        "m_re_values": m_values,
        "m_im": args.m_im,
        "shape": _shape_name(args.shape),
        "ay": args.ay,
        "az": args.az,
        "dpl": args.dpl,
        "kd": args.kd,
        "eps_exp": args.eps_exp,
        "target_residual": 10.0 ** (-args.eps_exp),
        "np": args.np,
        "maxiter": args.maxiter,
        "baseline_iter": "bicgstab",
        "precond_iter": args.precond_iter,
        "fgmres_restart": args.fgmres_restart if args.precond_iter == "fgmres" else None,
        "threshold_rel": args.threshold_rel,
        "max_radius": args.max_radius,
        "blend_identity": args.blend_identity,
        "symmetry": args.symmetry,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cases = [(grid, m_re) for grid in grids for m_re in m_values]
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]

    planned = 0
    for grid, m_re in cases:
        key = (grid, m_re)
        existing = rows_by_key.get(key)
        if existing and not args.force:
            existing_status = existing.get("status")
            if existing_status == "ok" or not args.retry_failed:
                print(f"skip existing grid={grid} m_re={m_re:g} status={existing_status}", flush=True)
                continue
        planned += 1
        row = _run_case(args, grid, m_re, out_dir)
        rows_by_key[key] = row
        _write_rows(rows_by_key, out_dir)
        _render_heatmaps(rows_by_key, out_dir)
        _summarize(rows_by_key, out_dir)

    if planned == 0:
        _write_rows(rows_by_key, out_dir)
        _render_heatmaps(rows_by_key, out_dir)
        _summarize(rows_by_key, out_dir)
        print("all requested cases were already present", flush=True)
    print(f"done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
