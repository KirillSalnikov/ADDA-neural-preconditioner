# Spectral ConvSAI Neural Preconditioner for ADDA

This branch is the standalone Spectral ConvSAI branch. It contains the spectral
neural preconditioner, its checkpoints, ADDA integration patches, and the
scripts needed to train, export, and benchmark it.

For the ConvSAI Universal / K2 v3 model, use the `convsai-universal` branch.

## What Is Included

- Model class: `neural_precond.model.ConvSAI_Spectral`
- Main checkpoint: `models/spectral/checkpoints/best_model.pt`
- Large-grid hex-prism checkpoint:
  `models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt`
- Exporter: `apps/export_spectral_precond.py`
- Real-ADDA evaluator: `apps/eval_spectral_adda.py`
- Heatmap benchmark: `apps/benchmark_spectral_heatmap_mpi.py`
- ADDA patch files: `adda_src_modified/*`

The large-grid checkpoint is about 1.84 MiB. It is the checkpoint used for the
current FGMRES heatmap.

## Build Patched ADDA

This repository does not vendor the whole upstream ADDA source tree. Start from
a clean upstream ADDA checkout and copy the patch files:

```bash
git clone https://github.com/adda-team/adda adda
cp adda_src_modified/* adda/src/

make -C adda/src seq mpi \
  FFTW3_INC_PATH="$HOME/.local/include" \
  FFTW3_LIB_PATH="$HOME/.local/lib"
```

If your workspace already has `adda/`, just repeat the `cp` and `make` steps.

## Export A Spectral Preconditioner

Spectral export is problem-specific. Re-export whenever shape, grid,
refractive index, or `dpl/kd` changes.

Example for hex prism `grid=80`, `m=2.5+0i`, `dpl=15`:

```bash
mkdir -p exports

python3 apps/export_spectral_precond.py \
  --checkpoint models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt \
  --shape prism \
  --ay 6.0 \
  --az 1.0 \
  --grid 80 \
  --m_re 2.5 \
  --m_im 0.0 \
  --kd 0.41887902047863906 \
  --threshold-rel 1e-6 \
  --max-radius 40 \
  --blend-identity 1.0 \
  --symmetry z180_zflip \
  --output exports/prism_g80_m25_spectral.precond
```

ADDA must use the matching `-grid`, `-shape`, `-m`, and `-dpl`. For `dpl=15`,
`kd = 2*pi/15 = 0.41887902047863906`.

## Run ADDA

Sequential:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

adda/src/seq/adda \
  -dir runs/prism_g80_m25_seq_spectral \
  -grid 80 \
  -m 2.5 0.0 \
  -shape prism 6.0 1.0 \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/prism_g80_m25_spectral.precond
```

MPI:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

mpirun -np 16 adda/src/mpi/adda_mpi \
  -dir runs/prism_g80_m25_mpi_spectral \
  -grid 80 \
  -m 2.5 0.0 \
  -shape prism 6.0 1.0 \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/prism_g80_m25_spectral.precond
```

Experimental FGMRES, useful for large FFTDIRECT/preconditioner-heavy cases:

```bash
ADDA_FGMRES_RESTART=100 \
mpirun -np 16 adda/src/mpi/adda_mpi \
  -dir runs/prism_g80_m25_mpi_spectral_fgmres \
  -grid 80 \
  -m 2.5 0.0 \
  -shape prism 6.0 1.0 \
  -dpl 15 \
  -eps 3 \
  -iter fgmres \
  -precond exports/prism_g80_m25_spectral.precond
```

## Import Optimization

For many-orientation runs (`-orient avg`), convert large mode-3 ConvSAI files to
cached FFTDIRECT mode 4 once, then reuse the converted file:

```bash
python3 apps/convert_convsai_to_fftdirect.py \
  --input exports/prism_g80_m25_spectral.precond \
  --output exports/prism_g80_m25_spectral.fftdirect.precond \
  --grid-x 160 \
  --grid-y 192 \
  --grid-z 160
```

Use ADDA's actual FFT grid dimensions from the ADDA `log` file. For import-heavy
MPI workflows, mode 5 x-slab float32 is also available:

```bash
python3 apps/convert_fftdirect_to_xslab_f32.py \
  --input exports/prism_g80_m25_spectral.fftdirect.precond \
  --output exports/prism_g80_m25_spectral.xslab_f32.precond
```

## Training

The large-grid checkpoint was trained with a real-ADDA promotion loop. The loop
starts from `best_hex_prism_real_32to96_r40.pt`, trains 25-step chunks, exports
each candidate with `--symmetry z180_zflip`, evaluates ADDA on grids 80 and 96,
and promotes only candidates with a better worst validation residual.

Exact launch command:

```bash
DEVICE=0 \
SEED_BASE=6200 \
CYCLES=200 \
STEPS_PER_CYCLE=25 \
LR=2e-6 \
TRAIN_GRID_MIN=32 \
TRAIN_GRID_MAX=96 \
VAL_GRIDS=80,96 \
SCORE_MODE=max \
M_RE=3.0 \
M_IM=0.0 \
DPL=15 \
KD=0.41887902047863906 \
RADIUS=40 \
LOSS=planewave_bicgstab \
KRYLOV_ITERS=1 \
ANCHOR_PROBE_WEIGHT=0.0 \
ANCHOR_RIGHT_PROBE_WEIGHT=0.0 \
SPECTRAL_FREQ_CHUNK_SIZE=65536 \
SPECTRAL_FREQ_CHECKPOINT_CHUNKS=1 \
CURRICULUM_FRAC=0.0 \
NP=16 \
VAL_MAXITER=120 \
VAL_TIMEOUT=600 \
EXPORT_SYMMETRY=z180_zflip \
START_CHECKPOINT=models/spectral/checkpoints/best_hex_prism_real_32to96_r40.pt \
BEST_CHECKPOINT=models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt \
BEST_SCORE_FILE=models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.score \
NAME_PREFIX=SPECTRAL_QUALITY10H_G32TO96_SYM_R40_20260525_100039 \
RUN_ROOT=runs/SPECTRAL_QUALITY10H_G32TO96_SYM_R40_20260525_100039 \
./train_spectral_hex_real_loop.sh
```

## Latest Heatmap

Run directory used locally:
`runs/HEATMAP_FGMRES_E3_20260531_115403`

Summary over 16 converged baseline+preconditioned cells:

- mean ADDA wall-time speedup: `9.93x`
- max ADDA wall-time speedup: `23.33x`
- mean total elapsed speedup including export: `4.43x`
- max total elapsed speedup including export: `15.87x`

Primary plots produced by `apps/benchmark_spectral_heatmap_mpi.py`:

- `heatmap_adda_wall_speedup.png`
- `heatmap_precond_wall_s.png`
- `heatmap_total_elapsed_speedup.png`

## File Layout

```text
models/spectral/                         Spectral checkpoints and model notes
apps/export_spectral_precond.py          Export Spectral checkpoint to .precond
apps/eval_spectral_adda.py               Real-ADDA checkpoint evaluation
apps/benchmark_spectral_heatmap_mpi.py   MPI heatmap benchmark
train_spectral_hex_real_loop.sh          Real-ADDA promotion training loop
train_v7/train.py                        Shared trainer with --spectral support
neural_precond/model.py                  ConvSAI_Spectral implementation
neural_precond/loss.py                   Training losses
adda_src_modified/                       Files copied into upstream ADDA
```
