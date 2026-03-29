# K² v3 — Свёрточный AI-прекондиционер для ADDA

## Что это

Нейросеть (13.7M параметров), которая предсказывает свёрточное ядро прекондиционера для ускорения итерационного решателя BiCGStab в ADDA.

**K²** означает, что ядро возводится в квадрат в частотной области: `M_hat = K_hat @ K_hat`. Это удваивает эффективный радиус (с 7 до 14 клеток) без дополнительных параметров.

Модель **универсальная** — один файл весов для любых форм (сфера, куб, эллипсоид, цилиндр, капсула) и любых параметров (m, kd, grid).

## Результаты в ADDA

| Конфигурация | Без прекондиционера | С K² v3 | Ускорение |
|---|:---:|:---:|:---:|
| Сфера g33 m=3.0 | 12 676 итер | 419 | **30x** |
| Сфера g33 m=3.5 | 20 001 | 626 | **32x** |
| Эллипсоид g48 m=3.5 | 20 000 | 373 | **54x** |
| Куб g33 m=3.0 | 12 676 | 418 | **30x** |

## Архитектура

```
Входы: m_re, m_im, kd, log(grid) + occupancy_grid (форма частицы)
  ↓
ShapeEncoder3D (3D CNN) → 16-dim вектор формы
  ↓
MLP: 20 → 512 → 512 → 512 → 25542 (r_cut=7, 1419 стенсильных точек × 18)
  ↓
Ядро свёртки K: (1419, 3, 3) complex
  ↓
FFT → K_hat @ K_hat → M_hat (прекондиционер в частотной области)
```

## Обучение

```bash
python train_v7/train.py \
    --name universal_r7_k2_v3 --device 0 --save \
    --loss adversarial --squared_kernel \
    --r_cut 7 --hidden_size 512 --num_layers 4 \
    --grid_min 8 --grid_max 64 \
    --ema_decay 0.999 --warmup_steps 500 --num_steps 60000 \
    --curriculum_frac 0.2 --lr 5e-4
```

Время обучения: ~8 часов на RTX 3090 Ti.

## Экспорт для ADDA

```bash
python apps/export_universal_precond.py \
    --checkpoint models/k2v3/checkpoints/best_model.pt \
    --squared_kernel \
    --shape sphere --grid 33 \
    --m_re 3.0 --m_im 0.0 --kd 0.4189 \
    --output /tmp/precond.precond
```

**Важно:** `--kd` должен совпадать с `2*pi/dpl`. При dpl=15: kd = 0.4189.

## Запуск ADDA

```bash
LD_LIBRARY_PATH=$HOME/.local/lib \
adda/src/seq/adda \
    -grid 33 -m 3.0 0.0 -shape sphere \
    -iter bicgstab -eps 5 -dpl 15 \
    -precond /tmp/precond.precond
```

**Использовать `-grid N`, НЕ `-size X`** — иначе dpl будет неточным.

## Файлы

- `checkpoints/best_model.pt` — обученные веса (53 MB)
- `../../train_v7/train.py` — скрипт обучения
- `../../apps/export_universal_precond.py` — скрипт экспорта
- `../../neural_precond/model.py` — класс `ConvSAI_Universal`
- `../../adda/src/precond.c` — C код применения в ADDA

## Преимущества и ограничения

**Плюсы:**
- Один файл весов для всех задач (не нужен переэкспорт)
- Компактный стенсиль (~10K записей, файл ~1.6 MB)
- Проверен на сотнях конфигураций в ADDA

**Минусы:**
- Эффективный радиус 14 — на больших grid (>48) ослабевает
- 13.7M параметров
