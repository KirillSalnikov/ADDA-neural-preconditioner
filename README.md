# ConvSAI Universal / K2 v3 Neural Preconditioner for ADDA

This branch is the standalone ConvSAI Universal branch, also called K2 v3. It
contains the reusable spatial convolution neural preconditioner, its checkpoint,
ADDA integration patches, and the scripts needed to train, export, and run it.

For the frequency-domain Spectral ConvSAI model, use the `spectral-convsai`
branch.

## What Is Included

- Model class: `neural_precond.model.ConvSAI_Universal`
- Checkpoint: `models/k2v3/checkpoints/best_model.pt`
- Exporter: `apps/export_universal_precond.py`
- ADDA patch files: `adda_src_modified/*`
- Optional import optimizers:
  `apps/convert_convsai_to_fftdirect.py`,
  `apps/convert_fftdirect_to_xslab_f32.py`

The checkpoint is about 52 MiB. The model predicts a translation-invariant
spatial convolution kernel from the particle occupancy grid and physical
parameters.

## Model

Inputs:

- occupancy grid of the particle shape;
- refractive index `m_re`, `m_im`;
- `kd`, where `kd = 2*pi/dpl`;
- `log(grid)`.

The network encodes the shape with a small 3D CNN and predicts a complex
3-by-3 convolution stencil. K2 mode squares the predicted kernel in the
frequency domain:

```text
M_hat = K_hat @ K_hat
```

That increases the effective radius without increasing the stored neural
network size.

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

## Export ConvSAI Universal To ADDA

Example for a sphere, `grid=48`, `m=3+0i`, `dpl=15`:

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
  --output exports/sphere_g48_m3_k2v3.precond
```

For prism-like shapes:

```bash
python3 apps/export_universal_precond.py \
  --checkpoint models/k2v3/checkpoints/best_model.pt \
  --squared_kernel \
  --shape prism \
  --ay 6.0 \
  --az 1.0 \
  --grid 80 \
  --m_re 2.5 \
  --m_im 0.0 \
  --kd 0.41887902047863906 \
  --output exports/prism_g80_m25_k2v3.precond
```

ADDA must use the matching `-grid`, `-shape`, `-m`, and `-dpl`. For `dpl=15`,
`kd = 2*pi/15 = 0.41887902047863906`.

## Run ADDA

Sequential:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

adda/src/seq/adda \
  -dir runs/sphere_g48_m3_seq_k2v3 \
  -grid 48 \
  -m 3.0 0.0 \
  -shape sphere \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/sphere_g48_m3_k2v3.precond
```

MPI:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

mpirun -np 16 adda/src/mpi/adda_mpi \
  -dir runs/sphere_g48_m3_mpi_k2v3 \
  -grid 48 \
  -m 3.0 0.0 \
  -shape sphere \
  -dpl 15 \
  -eps 3 \
  -iter bicgstab \
  -precond exports/sphere_g48_m3_k2v3.precond
```

## Import Optimization

For repeated-orientation workflows such as `-orient avg`, ADDA loads
`-precond` once before the orientation loop. If startup import dominates,
convert a mode-3 ConvSAI file to cached FFTDIRECT mode 4:

```bash
python3 apps/convert_convsai_to_fftdirect.py \
  --input exports/prism_g80_m25_k2v3.precond \
  --output exports/prism_g80_m25_k2v3.fftdirect.precond \
  --grid-x 160 \
  --grid-y 192 \
  --grid-z 160
```

Use ADDA's actual FFT grid dimensions from the ADDA `log` file. For import-heavy
MPI workflows, mode 5 x-slab float32 is also available:

```bash
python3 apps/convert_fftdirect_to_xslab_f32.py \
  --input exports/prism_g80_m25_k2v3.fftdirect.precond \
  --output exports/prism_g80_m25_k2v3.xslab_f32.precond
```

Then pass the converted `.precond` file to ADDA in the same way:

```bash
mpirun -np 16 adda/src/mpi/adda_mpi ... \
  -precond exports/prism_g80_m25_k2v3.fftdirect.precond
```

## Training

Typical K2 v3 training command:

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

Resume from an existing checkpoint:

```bash
python3 train_v7/train.py \
  --resume models/k2v3/checkpoints/best_model.pt \
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

## Expected Behavior

K2 v3 is compact at export time and produces smaller `.precond` files than the
Spectral branch. It is a good reusable model for moderate grids and varied
shapes. On very large grids and hard refractive indices, the Spectral branch is
usually stronger because it predicts directly in the FFT domain.

## File Layout

```text
models/k2v3/                         ConvSAI Universal checkpoint and notes
apps/export_universal_precond.py     Export K2 v3 checkpoint to .precond
apps/convert_convsai_to_fftdirect.py Optional mode-3 to mode-4 converter
train_v7/train.py                    ConvSAI Universal trainer
neural_precond/model.py              ConvSAI_Universal implementation
neural_precond/loss.py               Training losses
adda_src_modified/                   Files copied into upstream ADDA
docs/k2v3_guide.md                   Detailed K2 v3 notes
```
