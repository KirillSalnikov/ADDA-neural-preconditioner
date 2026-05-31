#!/usr/bin/env python3
"""Evaluate spectral checkpoints with real ADDA MPI runs.

The training loss is only a proxy. This script exports each checkpoint to an
ADDA preconditioner, runs ADDA on the target case, and ranks checkpoints by
the best actual residual reached in the ADDA log.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "apps" / "export_spectral_precond.py"


def _parse_residuals(text):
    return [(int(i), float(v))
            for i, v in re.findall(r"RE_(\d+)\s*=\s*([0-9.Ee+-]+)", text)]


def _shape_args(shape, ay, az):
    if shape == "prism":
        return ["-shape", "prism", str(int(ay)), str(az)]
    if shape == "sphere":
        return ["-shape", "sphere"]
    if shape == "cube":
        return ["-shape", "box"]
    if shape == "ellipsoid":
        return ["-shape", "ellipsoid", str(ay), str(az)]
    raise ValueError(f"unsupported shape for ADDA eval: {shape}")


def _run(cmd, timeout, env=None):
    try:
        r = subprocess.run(
            cmd, cwd=ROOT, env=env, timeout=timeout,
            capture_output=True, text=True,
        )
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = ""
        if e.stdout:
            out += e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else e.stdout
        if e.stderr:
            out += e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else e.stderr
        return 124, out + "\nTIMEOUT\n"


def _safe_name(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path))


def evaluate_checkpoint(args, checkpoint, grid, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    precond = case_dir / "precond.precond"

    export_cmd = [
        sys.executable, str(EXPORT),
        "--checkpoint", str(checkpoint),
        "--shape", args.shape,
        "--ay", str(args.ay),
        "--az", str(args.az),
        "--grid", str(grid),
        "--m_re", str(args.m_re),
        "--m_im", str(args.m_im),
        "--kd", str(args.kd),
        "--threshold-rel", str(args.threshold_rel),
        "--blend-identity", str(args.blend_identity),
        "--symmetry", str(args.symmetry),
        "--output", str(precond),
    ]
    if args.max_radius is not None:
        export_cmd += ["--max-radius", str(args.max_radius)]
    if args.normalize_inputs:
        export_cmd.append("--normalize-inputs")
    if args.spectral_correction_scale is not None:
        export_cmd += ["--spectral_correction_scale", str(args.spectral_correction_scale)]

    t0 = time.perf_counter()
    rc_export, export_out = _run(export_cmd, args.export_timeout)
    export_time = time.perf_counter() - t0
    (case_dir / "export.log").write_text(export_out, encoding="utf-8", errors="replace")
    if rc_export != 0 or not precond.exists():
        return {
            "checkpoint": str(checkpoint),
            "grid": grid,
            "ok": False,
            "stage": "export",
            "export_rc": rc_export,
            "error": export_out[-800:],
        }

    env = os.environ.copy()
    fftw_lib = os.path.expanduser(args.fftw_lib_path)
    env["LD_LIBRARY_PATH"] = f"{fftw_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    out_dir = case_dir / "adda"
    out_dir.mkdir(exist_ok=True)
    adda_cmd = [
        "mpirun", "--oversubscribe", "-np", str(args.np),
        "-x", f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}",
        args.adda_bin,
        "-grid", str(grid),
        "-m", str(args.m_re), str(args.m_im),
        *_shape_args(args.shape, args.ay, args.az),
        "-iter", "bicgstab",
        "-eps", str(args.eps_exp),
        "-dpl", str(args.dpl),
        "-maxiter", str(args.maxiter),
        "-dir", str(out_dir),
        "-precond", str(precond),
    ]

    t0 = time.perf_counter()
    rc_adda, adda_out = _run(adda_cmd, args.timeout, env=env)
    adda_time = time.perf_counter() - t0
    (case_dir / "adda.out").write_text(adda_out, encoding="utf-8", errors="replace")

    vals = _parse_residuals(adda_out)
    min_iter, min_res = (None, None)
    if vals:
        min_iter, min_res = min(vals, key=lambda x: x[1])
    last_iter, last_res = vals[-1] if vals else (None, None)
    converged = bool(vals and vals[-1][1] <= 10.0 ** (-args.eps_exp))

    return {
        "checkpoint": str(checkpoint),
        "grid": grid,
        "ok": True,
        "stage": "adda",
        "export_rc": rc_export,
        "adda_rc": rc_adda,
        "export_time_s": export_time,
        "adda_time_s": adda_time,
        "precond_bytes": precond.stat().st_size,
        "num_residuals": len(vals),
        "last_iter": last_iter,
        "last_residual": last_res,
        "min_iter": min_iter,
        "min_residual": min_res,
        "converged": converged,
        "case_dir": str(case_dir),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", default=[],
                   help="Checkpoint path. Can be repeated.")
    p.add_argument("--glob", action="append", default=[],
                   help="Glob for checkpoint paths. Can be repeated.")
    p.add_argument("--grid", type=int, action="append", required=True,
                   help="Grid to evaluate. Can be repeated.")
    p.add_argument("--shape", default="prism",
                   choices=["prism", "sphere", "cube", "ellipsoid"])
    p.add_argument("--ay", type=float, default=6.0)
    p.add_argument("--az", type=float, default=1.0)
    p.add_argument("--m_re", type=float, default=3.0)
    p.add_argument("--m_im", type=float, default=0.0)
    p.add_argument("--dpl", type=float, default=15.0)
    p.add_argument("--kd", type=float, default=0.41887902047863906)
    p.add_argument("--threshold-rel", type=float, default=1e-6)
    p.add_argument("--max-radius", type=int, default=32)
    p.add_argument("--blend-identity", type=float, default=1.0)
    p.add_argument("--symmetry", default="none",
                   choices=["none", "z180", "zflip", "z180_zflip", "d2"],
                   help="Average exported spatial kernel over cheap prism symmetries.")
    p.add_argument("--normalize-inputs", action="store_true")
    p.add_argument("--spectral-correction-scale", "--spectral_correction_scale",
                   dest="spectral_correction_scale", type=float, default=None,
                   help="Override additive spectral correction scale during export.")
    p.add_argument("--np", type=int, default=16)
    p.add_argument("--maxiter", type=int, default=180)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--export-timeout", type=float, default=300.0)
    p.add_argument("--eps-exp", type=int, default=5,
                   help="ADDA -eps exponent; 5 means residual target 1e-5.")
    p.add_argument("--output-dir", default="runs/spectral_real_eval")
    p.add_argument("--best-output", default=None,
                   help="Copy best checkpoint here, based on min residual.")
    p.add_argument("--adda-bin", default="adda/src/mpi/adda_mpi")
    p.add_argument("--fftw-lib-path", default="~/.local/lib")
    args = p.parse_args()

    checkpoints = [Path(c) for c in args.checkpoint]
    for pattern in args.glob:
        checkpoints.extend(Path(x) for x in sorted(glob.glob(pattern)))
    checkpoints = list(dict.fromkeys(checkpoints))
    if not checkpoints:
        raise SystemExit("no checkpoints provided")

    out_root = ROOT / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for checkpoint in checkpoints:
        for grid in args.grid:
            case_name = f"g{grid}_{_safe_name(checkpoint)}"
            print(f"==> {checkpoint} grid={grid}", flush=True)
            r = evaluate_checkpoint(args, checkpoint, grid, out_root / case_name)
            results.append(r)
            if r.get("ok"):
                print(
                    f"    min={r['min_residual']}@{r['min_iter']} "
                    f"last={r['last_residual']}@{r['last_iter']} "
                    f"converged={r['converged']} time={r['adda_time_s']:.1f}s",
                    flush=True,
                )
            else:
                print(f"    failed at {r.get('stage')}: {r.get('error', '')[-200:]}", flush=True)

    summary_json = out_root / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with (out_root / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("checkpoint\tgrid\tmin_iter\tmin_residual\tlast_iter\tlast_residual\tconverged\tcase_dir\n")
        for r in results:
            f.write(
                f"{r.get('checkpoint')}\t{r.get('grid')}\t{r.get('min_iter')}\t"
                f"{r.get('min_residual')}\t{r.get('last_iter')}\t{r.get('last_residual')}\t"
                f"{r.get('converged')}\t{r.get('case_dir')}\n"
            )

    ok_results = [r for r in results if r.get("min_residual") is not None]
    if ok_results:
        best = min(ok_results, key=lambda r: r["min_residual"])
        print(
            f"BEST: {best['checkpoint']} grid={best['grid']} "
            f"min={best['min_residual']}@{best['min_iter']} case={best['case_dir']}",
            flush=True,
        )
        if args.best_output:
            dst = ROOT / args.best_output
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best["checkpoint"], dst)
            print(f"Copied best checkpoint to {dst}", flush=True)


if __name__ == "__main__":
    main()
