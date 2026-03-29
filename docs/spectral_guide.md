# Spectral Neural Preconditioner для ADDA — Полный гайд

## Что это такое

**Прекондиционер** — это вспомогательная операция, которая ускоряет итерационный решатель (BiCGStab) при решении системы линейных уравнений в методе дискретных диполей (DDA/ADDA). Без прекондиционера решатель может делать тысячи итераций. С хорошим прекондиционером — сотни или десятки.

**Spectral Neural Preconditioner** — нейросеть, которая для каждой конкретной задачи (форма частицы, показатель преломления, размер сетки) предсказывает прекондиционер напрямую в частотной области. Она "смотрит" на матрицу задачи D_hat и предсказывает приближённую обратную матрицу M_hat.

### Результаты (ADDA, BiCGStab)

| Конфигурация | Без прекондиционера | Spectral | Ускорение | K² v3 (предыдущий лучший) |
|---|:---:|:---:|:---:|:---:|
| Сфера g24 m=3.0 | 4324 итер | 160 | **27x** | 197 (22x) |
| Сфера g33 m=3.5 | 20001 | 413 | **48x** | 626 (32x) |
| Сфера g48 m=3.0 | 20000 | 448 | **45x** | 20000 (FAIL!) |
| Куб g24 m=3.5 | 12590 | 415 | **30x** | 561 (22x) |

Параметры модели: **228K** (в 60 раз меньше чем K² v3 с 13.7M).

---

## Словарь терминов

- **DDA (Discrete Dipole Approximation)** — метод дискретных диполей. Частица разбивается на маленькие кубики (диполи), каждый из которых поляризуется в электромагнитном поле. Взаимодействие диполей описывается системой линейных уравнений.

- **ADDA** — программа на C для решения задач DDA. Использует FFT для быстрого вычисления матрично-векторного произведения.

- **BiCGStab** — итерационный метод решения систем линейных уравнений (Bi-Conjugate Gradient Stabilized). На каждой итерации вычисляет A·v (произведение матрицы на вектор) и постепенно приближается к решению.

- **Прекондиционер M** — приближённая обратная матрица: M ≈ A⁻¹. Если M·A близка к единичной матрице, BiCGStab сходится за мало итераций. Применяется как "левый прекондиционер": вместо A·x = b решаем M·A·x = M·b.

- **Grid (сетка)** — число диполей вдоль каждой оси. Grid=24 означает 24×24×24 кубиков. Чем больше grid, тем точнее, но тяжелее задача.

- **m (показатель преломления)** — комплексное число, характеризующее материал частицы. m=3.0 — сильно преломляющий материал (трудная задача). m=1.5 — лёгкая задача.

- **kd (параметр размера)** — безразмерный параметр = (2π/λ)·d, где λ — длина волны, d — размер одного диполя. Обычно kd ≈ 0.42 (при dpl=15).

- **dpl (dipoles per lambda)** — число диполей на длину волны. dpl=15 — стандартное значение. kd = 2π/dpl.

- **D_hat** — матрица взаимодействия диполей в частотной области (после Фурье-преобразования функции Грина). Содержит всю физику задачи. Размер: 6 компонент × (2·grid)³ комплексных чисел.

- **M_hat** — прекондиционер в частотной области. 3×3 комплексная матрица для каждой частотной точки. Применяется через FFT-свёртку.

- **FFT (Fast Fourier Transform)** — быстрое преобразование Фурье. Переводит данные из "пространственного" представления в "частотное" и обратно. Стоимость O(N log N).

- **Стенсиль (stencil)** — набор смещений (dx, dy, dz) и соответствующих 3×3 блоков. Компактное представление пространственного ядра свёртки. Прекондиционер хранится как стенсиль и применяется через FFT.

- **K² (squared kernel)** — приём удвоения эффективного радиуса: M_hat = K_hat @ K_hat (матричное произведение 3×3 для каждой частоты). Если K имеет радиус r, то K² имеет радиус 2r без дополнительных параметров.

- **Occupancy grid** — бинарная 3D сетка (0 или 1), показывающая какие ячейки заняты частицей. Подаётся в ShapeEncoder3D.

- **ShapeEncoder3D** — маленькая свёрточная нейросеть (3D CNN), которая превращает occupancy grid в компактный вектор (16 чисел), описывающий форму частицы.

- **Adversarial probe loss** — функция потерь для обучения. Ищет "наихудший" вектор z (через power iteration), для которого M·A·z максимально отличается от z. Затем минимизирует эту разницу. Это заставляет M быть хорошим прекондиционером для самых трудных направлений.

- **EMA (Exponential Moving Average)** — скользящее среднее весов модели. Сглаживает шум обучения. EMA модель часто лучше "сырой" модели.

- **Curriculum learning** — постепенное усложнение задач при обучении. Начинаем с маленьких grid (быстро), потом увеличиваем до больших (медленно, но важно для обобщения).

---

## Архитектура модели

```
ВХОДЫ:
  1. Физические параметры: m_re, m_im, kd, log(grid) — 4 числа
  2. Occupancy grid: бинарная 3D сетка формы частицы
  3. D_hat: матрица взаимодействия в частотной области (из FFTMatVec)

МОДЕЛЬ:
  ┌──────────────────────────────────┐
  │ ShapeEncoder3D (3D CNN)          │
  │ Occupancy grid → 16-dim вектор  │
  │ Conv3d(1→16) + Pool             │
  │ Conv3d(16→32) + Pool            │
  │ Conv3d(32→64) + Pool            │
  │ AdaptiveAvgPool + Linear → 16   │
  └───────────┬──────────────────────┘
              │
  ┌───────────▼──────────────────────┐
  │ Global Encoder MLP               │
  │ [m_re, m_im, kd, log_grid,      │
  │  shape_embed] = 20 чисел         │
  │ → 256 → ReLU → 256 → ReLU       │
  │ → 256 (conditioning vector)      │
  └───────────┬──────────────────────┘
              │
  ┌───────────▼──────────────────────┐
  │ Per-Frequency MLP                │
  │ Для КАЖДОЙ частотной точки k:    │
  │                                  │
  │ input = [Re(D_hat(k)) — 9 чисел,│
  │          Im(D_hat(k)) — 9 чисел, │
  │          kx/gx, ky/gy, kz/gz,   │
  │          conditioning — 256]     │
  │ = 277 чисел                      │
  │                                  │
  │ → 256 → ReLU → 256 → ReLU       │
  │ → 256 → ReLU → 18 (residual)    │
  │                                  │
  │ M_hat(k) = I + reshape(output)   │
  └──────────────────────────────────┘

  K²: M_hat = M_hat @ M_hat (3×3 на каждой частоте)

ВЫХОД: M_hat — (3, 3, 2·grid, 2·grid, 2·grid) complex
```

Ключевые свойства:
- **228K параметров** (в 60 раз меньше чем K² v3)
- **Одна и та же MLP** применяется к каждой из (2·grid)³ частотных точек
- **Нет ограничения по дальности** — работает на любом grid
- **Видит матрицу задачи** — адаптируется к конкретной физике через D_hat

---

## Файлы проекта

```
neural_precond/model.py        — класс ConvSAI_Spectral (и другие модели)
neural_precond/loss.py         — функции потерь (adversarial probe и др.)
train_v7/train.py              — скрипт обучения
neuralif/fft_matvec.py         — Python реализация матрично-векторного произведения (A·v)
apps/export_spectral_precond.py — экспорт M_hat → стенсиль → .precond файл
apps/export_universal_precond.py — экспорт стенсильных моделей (K² v3 и др.)
adda/src/precond.c             — C код загрузки и применения прекондиционера в ADDA
adda/src/precond.h             — заголовочный файл
test_spectral_adda.py          — тест spectral vs K²v3 в ADDA
krylov/bicgstab.py             — Python реализация BiCGStab
```

---

## Обучение: пошаговая инструкция

### 1. Подготовка окружения

```bash
# Python 3.10+, PyTorch 2.0+, CUDA
pip install torch numpy matplotlib

# Сборка ADDA (нужен FFTW3)
cd adda/src
make seq FFTW3_INC_PATH=$HOME/.local/include FFTW3_LIB_PATH=$HOME/.local/lib
```

### 2. Запуск обучения

```bash
python train_v7/train.py \
    --name spectral_v3_adda \
    --device 0 \
    --save \
    --loss adversarial \
    --spectral \
    --squared_kernel \
    --freq_hidden 256 \
    --freq_layers 5 \
    --global_hidden 256 \
    --global_layers 3 \
    --grid_min 8 \
    --grid_max 64 \
    --ema_decay 0.999 \
    --warmup_steps 500 \
    --num_steps 60000 \
    --curriculum_frac 0.25 \
    --lr 1e-3
```

**Что означает каждый параметр:**

| Параметр | Значение | Пояснение |
|----------|----------|-----------|
| `--spectral` | — | Включает spectral архитектуру (вместо стенсильной) |
| `--squared_kernel` | — | Включает K²: M_hat = M_hat @ M_hat |
| `--freq_hidden 256` | 256 | Ширина per-frequency MLP (больше = мощнее, но медленнее) |
| `--freq_layers 5` | 5 | Глубина per-frequency MLP (включая выходной слой) |
| `--global_hidden 256` | 256 | Ширина Global Encoder MLP |
| `--global_layers 3` | 3 | Глубина Global Encoder MLP |
| `--grid_min 8` | 8 | Минимальный размер сетки при обучении |
| `--grid_max 64` | 64 | Максимальный размер сетки |
| `--curriculum_frac 0.25` | 0.25 | Первые 25% обучения — постепенное увеличение grid от min до max |
| `--lr 1e-3` | 0.001 | Скорость обучения (learning rate) |
| `--ema_decay 0.999` | 0.999 | Коэффициент EMA (0 = выкл, 0.999 = рекомендуется) |
| `--warmup_steps 500` | 500 | Линейный разогрев LR от 0 до lr за первые 500 шагов |
| `--num_steps 60000` | 60000 | Общее число шагов обучения |
| `--loss adversarial` | — | Adversarial probe loss (лучшая функция потерь) |
| `--device 0` | 0 | Номер GPU (0 = первый) |
| `--save` | — | Сохранять чекпоинты |

### 3. Что происходит при обучении

На каждом шаге:

1. **Генерация случайной задачи:**
   - Случайный m_re из [1.5, 4.0] (с приоритетом сложных случаев m > 2.75)
   - Случайный m_im из [0.0, 0.5]
   - Случайный kd из [0.2, 0.8] (логарифмическое распределение)
   - Случайный grid из [grid_min, текущий grid_max]
   - Случайная форма: сфера(15%), эллипсоид(40%), куб(15%), цилиндр(15%), капсула(15%)
   - Для эллипсоидов: случайные пропорции осей

2. **Построение физики:**
   - Создание позиций диполей для выбранной формы
   - Построение FFTMatVec (матрица A через FFT)
   - Получение D_hat — матрицы взаимодействия в частотной области

3. **Forward pass нейросети:**
   - ShapeEncoder3D: occupancy grid → 16-dim вектор формы
   - Global Encoder: [m, kd, grid, shape] → 256-dim conditioning
   - Per-frequency MLP: для каждой частотной точки [D_hat(k), coords, cond] → M_hat(k)
   - K²: M_hat = M_hat @ M_hat

4. **Вычисление loss:**
   - Adversarial probe: найти худший вектор z через 10 итераций power iteration
   - loss = ||M·A·z - z||² / ||z||² на худших z
   - Чем меньше loss, тем лучше M ≈ A⁻¹

5. **Обновление весов:**
   - loss.backward() → вычисление градиентов
   - AdamW optimizer с gradient clipping (max norm = 1.0)
   - EMA обновление скользящего среднего весов

### 4. Мониторинг обучения

```bash
# Смотреть лог
tail -f results/spectral_v3_adda/train.log

# Ключевые метрики:
# loss — чем меньше, тем лучше (0.7-1.0 — хорошо)
# speedup — ускорение на валидации (BiCGStab iterations)
# "New best" — лучший результат сохраняется в best_model.pt
```

### 5. Время обучения

- GPU: RTX 3090 Ti (24 GB)
- Маленькие grid (8-20): ~50 steps/sec
- Большие grid (48-64): ~1-2 steps/sec
- 60K шагов ≈ 6-9 часов

---

## Экспорт и использование в ADDA

### Шаг 1: Экспорт прекондиционера для конкретной задачи

Spectral модель предсказывает M_hat для конкретной задачи (форма + m + kd + grid).
Поэтому экспорт нужно делать заново для каждой новой задачи.

```bash
python apps/export_spectral_precond.py \
    --checkpoint results/spectral_v3_adda/best_model.pt \
    --shape sphere \
    --grid 33 \
    --m_re 3.0 --m_im 0.0 \
    --kd 0.4189 \
    --output /tmp/precond_sphere_g33_m3.precond
```

**Что делает скрипт:**
1. Создаёт позиции диполей для указанной формы и grid
2. Строит FFTMatVec (получает D_hat)
3. Прогоняет нейросеть: D_hat → M_hat
4. Делает обратное FFT: M_hat → spatial kernel (стенсиль)
5. Сохраняет стенсиль в .precond файл (mode=3, формат CONVSAI)

**Важно:** `--kd` должен точно совпадать с `2*pi/dpl` который будет использован в ADDA.
При dpl=15: kd = 2*π/15 = 0.41888.

### Шаг 2: Запуск ADDA с прекондиционером

```bash
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda \
    -grid 33 \
    -m 3.0 0.0 \
    -shape sphere \
    -iter bicgstab \
    -eps 5 \
    -dpl 15 \
    -precond /tmp/precond_sphere_g33_m3.precond
```

**КРИТИЧЕСКИ ВАЖНО:**
- Используйте `-grid N -dpl 15`, а НЕ `-size X -dpl 15`
- `-size` заставляет ADDA подбирать grid, и реальный dpl может чуть отличаться от 15
- Это расхождение в dpl полностью убивает прекондиционер
- `-grid N -dpl 15` гарантирует точный dpl=15 и kd=2π/15

### Формат .precond файла

```
Header (40 байт):
  magic:    uint64 = 0x4E49464C
  n:        uint64 = 3 * N_dipoles (число DOF)
  n_stencil: uint64 = число записей в стенсиле
  mode:     uint64 = 3 (CONVSAI)
  reserved: uint64 = 0

Data:
  stencil:  n_stencil × 3 × int32  — смещения (dx, dy, dz)
  kernel:   n_stencil × 18 × float64 — 3×3 комплексные блоки
            (9 пар re,im для каждой записи)
```

ADDA при загрузке:
1. Читает стенсиль и kernel
2. Размещает kernel на FFT-сетке по смещениям стенсиля
3. Делает FFT → получает Phat (частотное представление)
4. При каждой итерации BiCGStab: scatter → FFT → multiply Phat → IFFT → gather

---

## Изменения в ADDA

### precond.h — добавлен mode=4

```c
#define PRECOND_MODE_FFTDIRECT 4
```

(Для прямой загрузки Phat без стенсиля — экспериментальный, пока не используется для spectral.)

### precond.c — добавлена функция DumpDhat

```c
void DumpDhat(const char *filename);
```

Экспортирует D_hat (матрицу Грина в частотной области) из ADDA.
Вызывается через `-dump_dhat <file>`.

Формат файла:
```
dims:  3 × uint64 (gridX, gridY, gridZ)
data:  6 × gridX × gridY × gridZ × 2 × float64
       (6 компонент верхнего треугольника, interleaved re/im)
```

### calculator.c — вызов DumpDhat

```c
// После InitDmatrix():
if (dump_dhat_filename!=NULL) DumpDhat(dump_dhat_filename);
```

### param.c — аргумент -dump_dhat

```c
PARSE_FUNC(dump_dhat)
{
    dump_dhat_filename=ScanStrError(argv[1],MAX_FNAME);
}
```

### Сборка

```bash
cd adda/src
make seq FFTW3_INC_PATH=$HOME/.local/include FFTW3_LIB_PATH=$HOME/.local/lib
```

---

## Важные детали для воспроизведения

### 1. Конвенция FFTMatVec

Python FFTMatVec **должен** использовать ADDA-совместимую конвенцию:

```python
# ПРАВИЛЬНО (совпадает с ADDA):
fft_mv = FFTMatVec(positions, k=1.0, m=complex(m_re, m_im), d=kd, device='cpu')

# НЕПРАВИЛЬНО (другая система, хотя A·v математически тот же):
fft_mv = FFTMatVec(positions, k=kd, m=complex(m_re, m_im), d=1.0, device='cpu')
```

Оба варианта дают одинаковый A·v (потому что kd = k*d одинаковый). Но D_hat отличается!
Spectral модель принимает D_hat на вход, поэтому конвенция D_hat должна совпадать между
обучением и экспортом.

В `train_v7/train.py`, функция `build_fft_matvec_from_positions`:
```python
def build_fft_matvec_from_positions(positions, m_re, m_im, kd, device):
    d = kd    # ADDA convention: d = kd, k = 1
    k = 1.0
    m = complex(m_re, m_im)
    return FFTMatVec(positions, k, m, d=d, device=device)
```

### 2. Identity init (нулевая инициализация)

Последний слой per-frequency MLP инициализируется нулями:
```python
nn.init.zeros_(last.weight)
nn.init.zeros_(last.bias)
```

Это гарантирует что на старте обучения M_hat = I (единичная матрица),
т.е. прекондиционер не влияет. Модель постепенно учится отклоняться от I.

С K²: I @ I = I, так что K² тоже стартует с identity.

### 3. Adversarial probe loss

```python
loss = ||M·A·z - z||² / ||z||²
```

где z — "наихудший" вектор, найденный через power iteration:

```
z₀ = random
for i in range(10):
    r = M·A·z - z        # невязка
    z = r / ||r||         # нормализация
```

Это находит направление z, для которого M·A наиболее далеко от I.
Оптимизируя loss на таких z, модель учится быть хорошим прекондиционером
для всех направлений, а не только для "средних".

### 4. Размер стенсиля при экспорте

Spectral модель работает в частотной области — у неё нет стенсиля.
При экспорте M_hat переводится в пространственную область через IFFT:

```python
M_spatial = np.fft.ifftn(M_hat, axes=(2,3,4))
```

Это даёт плотный стенсиль размером (2·grid)³. Для grid=48 это 884K записей
и файл ~135 MB. ADDA обрабатывает это нормально (делает FFT обратно при загрузке),
но экспорт занимает время.

### 5. LDR поляризуемость

Python и ADDA используют одинаковую формулу поляризуемости:
LDR (Lattice Dispersion Relation) по Draine & Goodman (1993):

```python
alpha_CM = (3/(4π)) * V * (m²-1)/(m²+2)
M = (B1 + (B2+B3·S)·m²)·kd² + i·(2/3)·kd³
alpha = alpha_CM / (1 - (alpha_CM/V)·M)
```

Реализовано в `apps/adda_matrix.py: ldr_polarizability()`.

---

## Ограничения и что можно улучшить

1. **Экспорт для каждой задачи**: нужно перезапускать export для каждой новой комбинации
   (shape, grid, m, kd). Стенсильный K² v3 — один файл для любого grid.

2. **Большие файлы**: стенсиль для grid=48 = 135 MB. Можно обрезать малые записи
   (threshold=1e-6), но это снижает качество.

3. **kd должен точно совпадать**: малейшее расхождение kd между обучением и ADDA
   убивает прекондиционер. Используйте `-grid N`, не `-size X`.

4. **Модель ещё дообучается**: текущий best 4.31x на validation (19K/60K шагов).
   Полностью обученная модель может быть ещё лучше.

5. **Не тестировалось на:**
   - m_im > 0 (поглощающие материалы)
   - несферические формы кроме куба и эллипсоида
   - grid > 64 (но архитектура масштабируется)

---

## Быстрый старт (копировать-вставить)

```bash
# 1. Обучение (6-9 часов на RTX 3090 Ti)
python train_v7/train.py --name my_spectral --device 0 --save \
    --loss adversarial --spectral --squared_kernel \
    --freq_hidden 256 --freq_layers 5 \
    --global_hidden 256 --global_layers 3 \
    --grid_min 8 --grid_max 64 \
    --ema_decay 0.999 --warmup_steps 500 --num_steps 60000 \
    --curriculum_frac 0.25 --lr 1e-3

# 2. Экспорт для конкретной задачи
python apps/export_spectral_precond.py \
    --checkpoint results/my_spectral/best_model.pt \
    --shape sphere --grid 33 --m_re 3.0 --kd 0.4189 \
    --output /tmp/my_precond.precond

# 3. Запуск ADDA
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda -grid 33 -m 3.0 0.0 -shape sphere \
    -iter bicgstab -eps 5 -dpl 15 \
    -precond /tmp/my_precond.precond
```
