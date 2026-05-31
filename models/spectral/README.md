# Spectral — Поточечный спектральный AI-прекондиционер для ADDA

## Что это

Нейросеть (228K параметров), которая для каждой частотной точки предсказывает оптимальный прекондиционер, видя матрицу задачи D_hat напрямую.

В отличие от стенсильных моделей, **Spectral не имеет ограничения по дальности** — одна и та же маленькая MLP применяется к каждой из (2·grid)³ частотных точек.

**K²** (squared): после предсказания M_hat применяется `M_hat = M_hat @ M_hat` для усиления.

## Результаты в ADDA

| Конфигурация | Без прекондиционера | Spectral | Ускорение | K² v3 (сравнение) |
|---|:---:|:---:|:---:|:---:|
| Сфера g24 m=3.0 | 4 324 итер | 160 | **27x** | 197 (22x) |
| Сфера g33 m=3.5 | 20 001 | 413 | **48x** | 626 (32x) |
| Сфера g48 m=3.0 | 20 000 | 448 | **45x** | 20 000 (FAIL!) |
| Куб g33 m=3.0 | 12 676 | 290 | **44x** | 418 (30x) |

Spectral побеждает K² v3 в **29 из 31** тестов. На g48 m=3.0 K² v3 не сходится — Spectral даёт 45x.

## Архитектура

```
Входы:
  1. Физические параметры: m_re, m_im, kd, log(grid)
  2. Occupancy grid → ShapeEncoder3D → 16-dim вектор формы
  3. D_hat: матрица взаимодействия в частотной области (из FFTMatVec)

Global Encoder MLP: [m, kd, grid, shape_embed] = 20 → 256 → 256 → 256 (conditioning)

Per-Frequency MLP (одна и та же для ВСЕХ частотных точек):
  input = [Re(D_hat(k)), Im(D_hat(k)), kx/gx, ky/gy, kz/gz, conditioning]
        = 18 + 3 + 256 = 277 чисел
  → 256 → 256 → 256 → 18 (residual)
  M_hat(k) = I + reshape(output, 3×3 complex)

K²: M_hat = M_hat @ M_hat
```

Параметры: ShapeEncoder3D 70K + Global Encoder 104K + Freq MLP 53K = **228K итого**.

## Обучение

```bash
python train_v7/train.py \
    --name spectral_v3_adda --device 0 --save \
    --loss adversarial --spectral --squared_kernel \
    --freq_hidden 256 --freq_layers 5 \
    --global_hidden 256 --global_layers 3 \
    --grid_min 8 --grid_max 64 \
    --ema_decay 0.999 --warmup_steps 500 --num_steps 60000 \
    --curriculum_frac 0.25 --lr 1e-3
```

Время обучения: ~9 часов на RTX 3090 Ti.

## Текущий large-grid checkpoint

Файл:

```text
models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt
```

Размер: около 1.84 MiB. Это Spectral ConvSAI для hex-prism, обученный на
grid 32..96, `m=3+0i`, `kd=2*pi/15`, с радиусом экспорта 40 и симметрией
`z180_zflip` при real-ADDA проверке.

Точная команда запуска обучения:

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

Скрипт обучает короткими блоками по 25 шагов, экспортирует кандидата в ADDA,
проверяет реальные запуски на `G80` и `G96`, и заменяет `BEST_CHECKPOINT`
только если худший residual на проверке стал меньше.

## Экспорт для ADDA

Spectral модель предсказывает M_hat для конкретной задачи. Экспорт нужен для каждой комбинации (форма, grid, m, kd).

```bash
python apps/export_spectral_precond.py \
    --checkpoint models/spectral/checkpoints/best_model.pt \
    --shape sphere --grid 33 \
    --m_re 3.0 --m_im 0.0 --kd 0.4189 \
    --output /tmp/precond.precond
```

Пример экспорта текущего large-grid checkpoint для hex-prism `G80, m=2.5+0i`:

```bash
mkdir -p exports

python3 apps/export_spectral_precond.py \
    --checkpoint models/spectral/checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt \
    --shape prism --ay 6.0 --az 1.0 \
    --grid 80 \
    --m_re 2.5 --m_im 0.0 \
    --kd 0.41887902047863906 \
    --threshold-rel 1e-6 \
    --max-radius 40 \
    --blend-identity 1.0 \
    --symmetry z180_zflip \
    --output exports/prism_g80_m25_quality10h_sym.precond
```

Что делает скрипт:
1. Создаёт позиции диполей для формы
2. Строит FFTMatVec (k=1.0, d=kd) → получает D_hat
3. Нейросеть: D_hat + параметры → M_hat
4. IFFT: M_hat → пространственное ядро → стенсиль
5. Записывает .precond файл (mode=3 CONVSAI)

## Запуск ADDA

```bash
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda \
    -grid 33 -m 3.0 0.0 -shape sphere \
    -iter bicgstab -eps 5 -dpl 15 \
    -precond /tmp/precond.precond
```

MPI пример импорта этого `.precond` в ADDA:

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"

mpirun -np 16 adda/src/mpi/adda_mpi \
    -dir runs/prism_g80_m25_mpi_quality10h_sym \
    -grid 80 \
    -m 2.5 0.0 \
    -shape prism 6.0 1.0 \
    -dpl 15 \
    -eps 3 \
    -iter bicgstab \
    -precond exports/prism_g80_m25_quality10h_sym.precond
```

Для больших задач можно пробовать экспериментальный FGMRES, который применяет
прекондиционер один раз на шаг Arnoldi:

```bash
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

### КРИТИЧЕСКИ ВАЖНО

- Используйте **`-grid N -dpl 15`** (точный kd = 2π/15)
- **НЕ** используйте **`-size X -dpl 15`** (kd будет неточным)
- Малейшее расхождение kd между обучением и ADDA **полностью убивает** прекондиционер

## Файлы

- `checkpoints/best_model.pt` — обученные веса (1.9 MB)
- `checkpoints/best_hex_prism_real_32to96_r40_quality10h_sym.pt` — текущий
  large-grid hex-prism checkpoint (1.9 MB)
- `../../train_v7/train.py` — скрипт обучения (с флагом `--spectral`)
- `../../apps/export_spectral_precond.py` — скрипт экспорта
- `../../neural_precond/model.py` — класс `ConvSAI_Spectral`
- `../../adda/src/precond.c` — C код применения в ADDA

## Преимущества и ограничения

**Плюсы:**
- 228K параметров (в 60 раз меньше K² v3)
- Нет ограничения по дальности — работает на любом grid
- Видит матрицу задачи D_hat — адаптируется к конкретной физике
- Побеждает K² v3 на больших grid и высоких m

**Минусы:**
- Нужен переэкспорт для каждой задачи (shape + grid + m + kd)
- Большие файлы стенсиля для grid ≥ 48 (~135 MB)
- kd должен точно совпадать с 2π/dpl
