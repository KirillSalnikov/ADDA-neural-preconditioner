# ADDA Neural Preconditioner

Neural preconditioners for accelerating the BiCGStab iterative solver in
[ADDA](https://github.com/adda-team/adda), a Discrete Dipole Approximation
(DDA) electromagnetic scattering code.

This repository contains the neural models, training/export scripts, and ADDA
preconditioner integration code. The current production path has two neural
preconditioner families:

1. `ConvSAI_Universal`, also called **ConvSAI Universal** or **K^2 v3**.
2. `ConvSAI_Spectral`, also called **Spectral ConvSAI** or **Spectral**.

Both models export an ADDA `.precond` file. ADDA loads the file with
`-precond <file>` and applies it as a left preconditioner. For ADDA, both
families usually appear as `mode=3 CONVSAI` files; the difference is how the
kernel was predicted before export.

## Model Families

### 1. ConvSAI Universal / K^2 v3

Code class: `neural_precond.model.ConvSAI_Universal`

Exporter: `apps/export_universal_precond.py`

Default checkpoint:
`models/k2v3/checkpoints/best_model.pt`

ConvSAI Universal predicts a translation-invariant spatial convolution kernel
from the particle occupancy grid and physical parameters:

- shape occupancy grid, encoded by a small 3D CNN;
- refractive index `m_re`, `m_im`;
- `kd`, where `kd = 2*pi/dpl` for ADDA runs using `-dpl`;
- `log(grid)`.

The best checkpoint is called **K^2 v3** because it exports the learned kernel
as `M_hat = K_hat @ K_hat` in the frequency domain. This squares the learned
kernel and increases the effective spatial radius without increasing the neural
network size.

Use this family when you want a compact reusable spatial model. It is usually
cheap to export and produces smaller `.precond` files than the full spectral
model.

### 2. Spectral ConvSAI / Spectral

Code class: `neural_precond.model.ConvSAI_Spectral`

Exporter: `apps/export_spectral_precond.py`

Default checkpoint:
`models/spectral/checkpoints/best_model.pt`

Large-grid hex-prism checkpoint used for the current ADDA heatmaps:
`models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt`

Spectral ConvSAI predicts the preconditioner directly in the FFT domain. For
each frequency point it sees:

- the local `D_hat(k)` interaction matrix;
- normalized frequency coordinates;
- global conditioning from shape and physical parameters.

The same small MLP is applied to every frequency point. The exported result is
converted back to a spatial stencil and written as an ADDA `mode=3 CONVSAI`
file. The spectral model has no fixed spatial radius during prediction, so it
is usually stronger on larger grids and harder refractive indices. The tradeoff
is that export must be repeated for each exact problem and the resulting
`.precond` file can be large.

## Current ADDA Support

The ADDA integration supports:

- sequential ADDA: `adda/src/seq/adda`;
- MPI ADDA: `adda/src/mpi/adda_mpi`;
- `.precond` modes:
  - `mode=1`: sparse SAI;
  - `mode=2`: polynomial preconditioner;
  - `mode=3`: ConvSAI FFT/direct convolution;
  - `mode=4`: FFTDIRECT, cached full frequency-domain ConvSAI kernel;
  - `mode=5`: FFTDIRECT x-slab float32, experimental MPI import format.

ConvSAI has two apply paths:

- small threshold-pruned stencils use direct sparse convolution by default
  (`n_stencil <= 8192`);
- larger stencils use FFT convolution.

Useful runtime switches:

```bash
ADDA_CONVSAI_DIRECT=0              # force FFT convolution
ADDA_CONVSAI_DIRECT=1              # force direct sparse convolution
ADDA_CONVSAI_DIRECT_MAX_STENCIL=8192
ADDA_CONVSAI_DIRECT_PRECOMPUTE=1
ADDA_CONVSAI_DIRECT_PRECOMPUTE_MB=256
```

For MPI FFT convolution, ConvSAI/FFTDIRECT use ADDA's distributed FFT layout by
default. That avoids the old slow behavior where every MPI rank performed the
full 3D FFT redundantly. The fallback is still available:

```bash
ADDA_CONVSAI_DISTRIBUTED=0 mpirun -np 8 adda/src/mpi/adda_mpi ...
```

Do not use that fallback for performance measurements unless you are debugging.

For `-orient avg`, ADDA loads `-precond` once before the orientation loop. The
expensive startup cost for large `mode=3` ConvSAI files is rebuilding the
frequency-domain kernel with 9 FFTs. Convert such files once to `mode=4`
FFTDIRECT to cache that kernel and make later imports much cheaper:

```bash
python3 apps/convert_convsai_to_fftdirect.py \
  --input exports/prism_g80_m25_spectral.precond \
  --output exports/prism_g80_m25_spectral.fftdirect.precond \
  --grid-x 160 \
  --grid-y 192 \
  --grid-z 160
```

Use ADDA's actual FFT grid dimensions from the ADDA `log` file, not just the
particle `-grid` value. Then pass the cached file normally:

```bash
mpirun -np 16 adda/src/mpi/adda_mpi ... \
  -orient avg orient_params.dat \
  -precond exports/prism_g80_m25_spectral.fftdirect.precond
```

An experimental `mode=5` format is also available for import-heavy workflows.
It converts a `mode=4` file to x-slab-major complex float32, so MPI ranks read
their local frequency-domain slab with one contiguous read and the file is about
2x smaller:

```bash
python3 apps/convert_fftdirect_to_xslab_f32.py \
  --input exports/prism_g80_m25_spectral.fftdirect.precond \
  --output exports/prism_g80_m25_spectral.xslab_f32.precond
```

ADDA converts `mode=5` values back to `doublecomplex` in memory. Local tests
showed that it loads correctly, but float32 storage can slightly change
iteration counts, so keep `mode=4` as the default accuracy/performance baseline
unless startup I/O dominates the run.

There is also an experimental periodic correction mode:

```bash
ADDA_PRECOND_PERIODIC_CORRECTION=100 \
ADDA_PRECOND_PERIODIC_RELAX=1.0 \
mpirun -np 16 adda/src/mpi/adda_mpi ... \
  -iter bicgstab \
  -precond exports/prism_g80_m25_spectral.fftdirect.precond
```

This disables left preconditioning and tries `x <- x + relax*M*r` every `K`
ordinary BiCGStab iterations, accepting the trial only if the true residual does
not get worse. It is useful for experiments with cheaper occasional neural
corrections, but current local G32 tests did not beat the normal `mode=4`
left-preconditioned solve.

For large FFTDIRECT neural preconditioners, an experimental right-preconditioned
FGMRES solver can reduce the number of expensive preconditioner applications:

```bash
ADDA_FGMRES_RESTART=100 \
mpirun -np 16 adda/src/mpi/adda_mpi ... \
  -iter fgmres \
  -precond exports/prism_g80_m25_spectral.fftdirect.precond
```

`-iter fgmres` applies `M` once per Arnoldi step, while the BiCGStab
left-preconditioned path applies `M` twice per iteration. Current local tests on
G80 favored `ADDA_FGMRES_RESTART=100` and reduced wall time from 83.97 s
(`bicgstab`) to 72.07 s. On small G32 cases the result is noisy and may not beat
BiCGStab, so use FGMRES primarily for larger grids. This is still experimental
and uses more memory for the Krylov basis.

To benchmark the FGMRES path over the spectral preconditioner heatmap:

```bash
python3 apps/benchmark_spectral_heatmap_mpi.py \
  --grids 32,48,64,80,96 \
  --m-re-values 1.5,2.0,2.5,3.0,3.5 \
  --precond-iter fgmres \
  --fgmres-restart 100 \
  --maxiter 20000 \
  --np 16 \
  --adda-timeout 3600 \
  --export-timeout 1800
```

The script writes speedup heatmaps and absolute wall-time heatmaps, including
`heatmap_adda_wall_speedup.png`, `heatmap_precond_wall_s.png`, and
`heatmap_total_elapsed_speedup.png`.

Recent local validation after the distributed MPI fix:

| Case | Baseline ADDA | Old MPI precond path | New MPI precond path |
|---|---:|---:|---:|
| sphere `g32`, `m=3` | 15.31 s | 20.10 s | 2.13 s |
| sphere `g48`, `m=3` | 108.26 s | 71.49 s | 8.09 s |
| sphere `g64`, `m=2` | 141.24 s | 476.15 s | 53.69 s |

## Requirements

Python side:

- Python 3.10+;
- PyTorch;
- NumPy;
- SciPy;
- torch-geometric, required by the model package;
- matplotlib, optional for plots/reports.

ADDA side:

- C compiler and Fortran compiler;
- MPI compiler wrapper, for `adda_mpi`;
- FFTW3 headers and library.

Example Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy matplotlib torch-geometric
```

If PyTorch/torch-geometric need CUDA-specific wheels, install them using the
commands from their official install pages for your CUDA version.

## Build ADDA

A clean checkout of this repository tracks the ADDA integration patch files,
not the whole upstream ADDA source tree. If your workspace already has a full
ADDA checkout under `adda/`, copy the current patch files into it and build:

```bash
cp adda_src_modified/* adda/src/

make -C adda/src seq mpi \
  FFTW3_INC_PATH="$HOME/.local/include" \
  FFTW3_LIB_PATH="$HOME/.local/lib" \
  EXTRA_FLAGS="-march=native -mtune=native -flto -fno-math-errno -fno-strict-aliasing -funroll-loops"
```

If you start from a clean upstream ADDA checkout:

```bash
git clone https://github.com/adda-team/adda adda
cp adda_src_modified/* adda/src/

make -C adda/src seq mpi \
  FFTW3_INC_PATH="$HOME/.local/include" \
  FFTW3_LIB_PATH="$HOME/.local/lib"
```

Runtime loader path for local FFTW3 builds:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
```

## Important Parameter Convention

For ADDA runs using `-dpl`, the exporter must use:

```text
kd = 2*pi/dpl
```

For the common value `-dpl 15`:

```text
kd = 0.41887902047863906
```

The exported `.precond` file must match the ADDA run:

- same shape;
- same grid;
- same `m_re`, `m_im`;
- same `kd`, therefore same `dpl`;
- use `-grid N`, not `-size`, when reproducing the exported geometry.

Small mismatches can destroy convergence.

ADDA's `-eps 3` means a relative residual target of about `1e-3`.

## Train ConvSAI Universal / K^2 v3

Typical training command:

```bash
python3 train_v7/train.py \
  --name convsai_universal_k2_v3 \
  --device 0 \
  --save \
  --loss adversarial \
  --squared_kernel \
  --r_cut 7 \
  --hidden_size 512 \
  --num_layers 4 \
  --grid_min 8 \
  --grid_max 64 \
  --m_re_min 1.5 \
  --m_re_max 4.0 \
  --kd_min 0.2 \
  --kd_max 0.8 \
  --ema_decay 0.999 \
  --warmup_steps 500 \
  --num_steps 60000 \
  --lr 5e-4
```

Previous good runs were in the 8 hour range on an RTX 3090 Ti, but runtime
depends strongly on validation settings and grid range.

Resume:

```bash
python3 train_v7/train.py \
  --resume results/<run_name>/latest_model.pt \
  --name convsai_universal_k2_v3_resume \
  --device 0 \
  --save \
  --loss adversarial \
  --squared_kernel \
  --r_cut 7 \
  --hidden_size 512 \
  --num_layers 4 \
  --grid_min 8 \
  --grid_max 64 \
  --num_steps 60000 \
  --lr 5e-4
```

## Train Spectral ConvSAI

General training command:

```bash
python3 train_v7/train.py \
  --name spectral_convsai_g8_64 \
  --device 0 \
  --save \
  --loss adversarial \
  --spectral \
  --squared_kernel \
  --freq_hidden 256 \
  --freq_layers 5 \
  --global_hidden 256 \
  --global_layers 3 \
  --shape_embed_dim 16 \
  --grid_min 8 \
  --grid_max 64 \
  --m_re_min 1.5 \
  --m_re_max 4.0 \
  --kd_min 0.2 \
  --kd_max 0.8 \
  --ema_decay 0.999 \
  --warmup_steps 500 \
  --num_steps 60000 \
  --lr 1e-3
```

For larger grids, reduce activation memory with frequency chunking:

```bash
python3 train_v7/train.py \
  --name spectral_convsai_g32_96 \
  --device 0 \
  --save \
  --loss adversarial \
  --spectral \
  --squared_kernel \
  --freq_hidden 256 \
  --freq_layers 5 \
  --global_hidden 256 \
  --global_layers 3 \
  --grid_min 32 \
  --grid_max 96 \
  --spectral_freq_chunk_size 65536 \
  --spectral_freq_checkpoint_chunks \
  --ema_decay 0.999 \
  --warmup_steps 500 \
  --num_steps 60000 \
  --lr 1e-3
```

The checked large-grid hex-prism checkpoint
`models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt`
was trained with a real-ADDA promotion loop. The loop starts from
`best_hex_prism_real_32to96_r40.pt`, trains 25-step chunks, exports each
candidate with `--symmetry z180_zflip`, evaluates ADDA on grids 80 and 96, and
keeps the candidate only when the worst validation residual improves.

Exact launch command used for that run:

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

Useful Spectral export-validation options during training:

```bash
--export_threshold_rel 1e-8
--export_max_radius <radius>
--export_identity_blend 1.0
--adda_val_np 8
--adda_mpi_bin adda/src/mpi/adda_mpi
--fftw_lib_path "$HOME/.local/lib"
```

## Export ConvSAI Universal / K^2 v3 to ADDA

Example for a sphere, grid 48, `m=3+0i`, `dpl=15`:

```bash
mkdir -p exports

python3 apps/export_universal_precond.py \
  --checkpoint models/k2v3/checkpoints/best_model.pt \
  --squared_kernel \
  --shape sphere \
  --grid 48 \
  --m_re 3.0 \
  --m_im 0.0 \
  --kd 0.41887902047863906 \
  --output exports/sphere_g48_m3_convsai_k2.precond
```

For ellipsoids, pass `--ay` and `--az`. For prism-like shapes, use the same
shape name and dimensions that the exporter supports through
`make_shape_positions()`.

## Export Spectral ConvSAI to ADDA

Example for the same problem:

```bash
mkdir -p exports

python3 apps/export_spectral_precond.py \
  --checkpoint models/spectral/checkpoints/best_model.pt \
  --shape sphere \
  --grid 48 \
  --m_re 3.0 \
  --m_im 0.0 \
  --kd 0.41887902047863906 \
  --threshold-rel 1e-8 \
  --blend-identity 1.0 \
  --output exports/sphere_g48_m3_spectral.precond
```

Optional symmetry averaging for compatible prism cases:

```bash
--symmetry z180_zflip
```

Spectral export is problem-specific. Re-export when shape, grid, refractive
index, or `dpl/kd` changes.

Example using the checked large-grid hex-prism checkpoint:

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
  --output exports/prism_g80_m25_quality10h_sym.precond
```

## Run ADDA Sequential

Baseline without a preconditioner:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

adda/src/seq/adda \
  -dir runs/sphere_g48_m3_seq_base \
  -grid 48 \
  -m 3.0 0.0 \
  -shape sphere \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab
```

With an exported preconditioner:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

adda/src/seq/adda \
  -dir runs/sphere_g48_m3_seq_pre \
  -grid 48 \
  -m 3.0 0.0 \
  -shape sphere \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/sphere_g48_m3_spectral.precond
```

The same command works for a ConvSAI Universal/K^2 `.precond` file; only the
path after `-precond` changes.

Checked large-grid hex-prism example:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

adda/src/seq/adda \
  -dir runs/prism_g80_m25_seq_quality10h_sym \
  -grid 80 \
  -m 2.5 0.0 \
  -shape prism 6.0 1.0 \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/prism_g80_m25_quality10h_sym.precond
```

## Run ADDA MPI

Baseline:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

mpirun -np 8 adda/src/mpi/adda_mpi \
  -dir runs/sphere_g48_m3_mpi_base \
  -grid 48 \
  -m 3.0 0.0 \
  -shape sphere \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab
```

With a preconditioner:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

mpirun -np 8 adda/src/mpi/adda_mpi \
  -dir runs/sphere_g48_m3_mpi_pre \
  -grid 48 \
  -m 3.0 0.0 \
  -shape sphere \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/sphere_g48_m3_spectral.precond
```

The distributed ConvSAI apply path is enabled by default in MPI. To verify that
the fast FFT path is used, ADDA stdout should include:

```text
distributed MPI apply
```

For small threshold-pruned ConvSAI files, ADDA may instead print:

```text
direct sparse convolution
```

That is also expected; it avoids FFT overhead when the exported stencil is small
enough.

For debugging only, force the old replicated full-grid path:

```bash
ADDA_CONVSAI_DISTRIBUTED=0 mpirun -np 8 adda/src/mpi/adda_mpi ...
```

For large FFTDIRECT/preconditioner-heavy cases, the experimental FGMRES path can
reduce preconditioner applications:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

ADDA_FGMRES_RESTART=100 \
mpirun -np 16 adda/src/mpi/adda_mpi \
  -dir runs/prism_g80_m25_mpi_quality10h_sym_fgmres \
  -grid 80 \
  -m 2.5 0.0 \
  -shape prism 6.0 1.0 \
  -dpl 15 \
  -eps 3 \
  -iter fgmres \
  -precond exports/prism_g80_m25_quality10h_sym.precond
```

## Inspect ADDA Results

ADDA writes a `log` file in the `-dir` directory. Useful lines:

```bash
rg "Total number of iterations|Total wall time|one iteration|File I/O" \
  runs/sphere_g48_m3_mpi_pre/log
```

A good preconditioner should reduce iterations enough to overcome the extra
cost of applying `M*v`. For MPI ConvSAI, compare wall time, not only iteration
count.

## File Layout

```text
models/k2v3/
  checkpoints/best_model.pt        ConvSAI_Universal / K^2 v3 checkpoint

models/spectral/
  checkpoints/best_model.pt        ConvSAI_Spectral checkpoint
  checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt
                                   Checked hex-prism 32..96 checkpoint

neural_precond/model.py            Model classes:
                                   ConvSAI_Universal, ConvSAI_Spectral,
                                   ConvSAI_Multigrid, legacy variants
neural_precond/loss.py             Probe/adversarial/Krylov losses
train_v7/train.py                  Main training script for both families

apps/export_universal_precond.py   Export ConvSAI_Universal/K^2 to .precond
apps/export_spectral_precond.py    Export ConvSAI_Spectral to .precond
apps/export_sai_precond.py         Sparse SAI exporter support
apps/adda_matrix.py                DDA geometry/matrix helpers

core/fft_matvec.py                 Python FFT MatVec used during training
core/models.py                     Base GraphNet/MLP blocks
krylov/bicgstab.py                 Python BiCGStab validation

adda/src/precond.c                 ADDA preconditioner load/apply code
adda/src/precond.h                 ADDA preconditioner data structures
adda_src_modified/                 Patch copy for upstream ADDA source trees

docs/k2v3_guide.md                 Detailed ConvSAI/K^2 guide
docs/spectral_guide.md             Detailed Spectral guide
```

## Naming Summary

Use these names consistently:

| Short name | Exact code class | Export script | Typical checkpoint |
|---|---|---|---|
| ConvSAI Universal / K^2 v3 | `ConvSAI_Universal` | `apps/export_universal_precond.py` | `models/k2v3/checkpoints/best_model.pt` |
| Spectral ConvSAI / Spectral | `ConvSAI_Spectral` | `apps/export_spectral_precond.py` | `models/spectral/checkpoints/best_model.pt` |

`ConvSAI_MLP` is the older non-universal spatial ConvSAI implementation.
`ConvSAI_Multigrid`, `ConvSAI_Separable`, and `ConvSAI_Hybrid` are experimental
variants. They are useful for research, but the two production paths above are
the ones to document and compare by default.

## Credits

- [ADDA](https://github.com/adda-team/adda), the DDA solver.
- [neural-incomplete-factorization](https://github.com/paulhausner/neural-incomplete-factorization),
  which inspired the learned preconditioner direction and provided useful base
  neural-network building blocks.
- [autoresearch](https://github.com/karpathy/autoresearch), used for automated
  experiment search during earlier model exploration.
