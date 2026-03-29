#!/usr/bin/env python3
"""Generate PDF guide for Spectral Neural Preconditioner — readable layout."""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10


def new_page(pdf, title, figsize=(11, 8.5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.97)
    return fig, ax


def add_text(ax, y, text, fontsize=10, bold=False, color='black', indent=0):
    weight = 'bold' if bold else 'normal'
    ax.text(0.04 + indent * 0.03, y, text, transform=ax.transAxes,
            fontsize=fontsize, va='top', fontweight=weight, color=color, wrap=True)


def add_code(ax, y, text, fontsize=8):
    ax.text(0.06, y, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', fontfamily='monospace', color='#1B5E20',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F1F8E9', alpha=0.9))


def add_warning(ax, y, text, fontsize=9):
    ax.text(0.06, y, text, transform=ax.transAxes, fontsize=fontsize,
            va='top', fontweight='bold', color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: Introduction
# ═══════════════════════════════════════════════════════════════════════
def page1_intro(pdf):
    fig, ax = new_page(pdf, 'Spectral Neural Preconditioner для ADDA')

    add_text(ax, 0.88, 'Что это такое', fontsize=14, bold=True)
    add_text(ax, 0.83,
        'Нейросеть (228 тысяч параметров), которая ускоряет итерационный решатель\n'
        'BiCGStab в программе ADDA (метод дискретных диполей, DDA).\n\n'
        'Прекондиционер M ≈ A⁻¹ — это приближённая обратная матрица.\n'
        'Если M хорош, BiCGStab сходится за десятки итераций вместо тысяч.\n\n'
        'Spectral модель уникальна: она видит матрицу задачи (D_hat)\n'
        'и подбирает оптимальный M для каждой конкретной задачи.')

    add_text(ax, 0.58, 'Результаты в ADDA', fontsize=14, bold=True)

    table_data = [
        ['Конфигурация', 'Без прекон.', 'Spectral', 'Ускорение', 'K² v3\n(прежний лучший)'],
        ['Сфера g24 m=3.0', '4 324', '160', '27x', '197 (22x)'],
        ['Сфера g33 m=3.5', '20 001', '413', '48x', '626 (32x)'],
        ['Сфера g48 m=3.0', '20 000', '448', '45x', '20 000 (FAIL!)'],
        ['Куб g24 m=3.5', '12 590', '415', '30x', '561 (22x)'],
        ['Куб g33 m=3.0', '12 676', '290', '44x', '418 (30x)'],
    ]
    t = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='upper center',
                 bbox=[0.04, 0.22, 0.92, 0.32])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for j in range(5):
        t[0, j].set_facecolor('#1565C0')
        t[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)):
        t[i, 3].set_facecolor('#E8F5E9')
        t[i, 3].set_text_props(fontweight='bold')

    add_text(ax, 0.18,
        'Spectral побеждает K² v3 в 29 из 31 протестированных случаев.\n'
        'Медианное ускорение: Spectral 18.6x  vs  K² v3 12.3x.\n'
        'Параметров: 228K (в 60 раз меньше чем K² v3 с 13.7M).',
        fontsize=10)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: Glossary
# ═══════════════════════════════════════════════════════════════════════
def page2_glossary(pdf):
    fig, ax = new_page(pdf, 'Словарь терминов')

    terms = [
        ('DDA', 'Метод дискретных диполей. Частица разбивается на кубики (диполи),\nих взаимодействие описывается системой линейных уравнений.'),
        ('ADDA', 'Программа на C для решения задач DDA. Использует FFT для быстрого A·v.'),
        ('BiCGStab', 'Итерационный метод решения Ax=b. На каждой итерации считает A·v\nи приближается к ответу. Меньше итераций → быстрее.'),
        ('Прекондиционер M', 'Приближение M ≈ A⁻¹. Задача A·x=b превращается в M·A·x = M·b,\nгде M·A ≈ I → быстрая сходимость.'),
        ('Grid (сетка)', 'Число диполей по каждой оси. Grid=24 → куб 24×24×24 диполей.'),
        ('m', 'Показатель преломления материала. m=3.0 — трудная задача, m=1.5 — лёгкая.'),
        ('kd', 'Безразмерный параметр = (2π/λ)·d.  При dpl=15: kd = 2π/15 ≈ 0.4189.'),
        ('D_hat', 'Матрица взаимодействия в частотной области (FFT функции Грина).\nСодержит всю физику. Размер: 6 × (2·grid)³ комплексных чисел.'),
        ('M_hat', 'Прекондиционер в частотной области. 3×3 комплексная матрица\nна каждой частоте. Применяется через FFT-свёртку (стоимость ≈ 1 A·v).'),
        ('K²', 'M_hat = K_hat @ K_hat — матричное произведение на каждой частоте.\nУдваивает эффективный радиус без доп. параметров.'),
        ('FFT', 'Быстрое преобразование Фурье. Переводит данные из пространственного\nв частотное представление. Стоимость O(N log N).'),
        ('Стенсиль', 'Набор смещений (dx,dy,dz) + блоки 3×3. Компактное хранение ядра свёртки.'),
        ('Adversarial\nprobe loss', 'Функция потерь: ищет наихудший вектор z и минимизирует\n||M·A·z − z||²/||z||². Гарантирует хороший M для всех направлений.'),
    ]

    y = 0.90
    for name, desc in terms:
        add_text(ax, y, f'  {name}', fontsize=9, bold=True, color='#1565C0')
        add_text(ax, y - 0.015, f'     {desc}', fontsize=8)
        lines = desc.count('\n') + 1
        y -= 0.025 + lines * 0.023

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: Architecture
# ═══════════════════════════════════════════════════════════════════════
def page3_architecture(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    fig.suptitle('Архитектура модели', fontsize=18, fontweight='bold', y=0.97)

    def box(x, y, w, h, text, color, fs=8):
        r = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.12",
                           facecolor=color, edgecolor='#333', lw=1.5)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
                fontweight='bold')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#555'))

    # Inputs
    box(0.2, 7.0, 2.2, 1.1, 'Occupancy Grid\n(бинарная 3D сетка\nформы частицы)', '#E3F2FD')
    box(2.9, 7.0, 2.2, 1.1, 'Физические параметры\nm_re, m_im, kd,\nlog(grid)', '#E8F5E9')
    box(6.5, 7.0, 3.0, 1.1, 'D_hat из FFTMatVec\n(матрица взаимодействия\nв частотной области)', '#FFF3E0')

    # ShapeEncoder
    box(0.2, 5.4, 2.2, 1.2, 'ShapeEncoder3D\n3D свёрточная сеть\nConv3d×3 + Pool×3\n→ 16 чисел', '#BBDEFB', 8)
    arrow(1.3, 7.0, 1.3, 6.6)

    # Concat
    box(1.0, 4.5, 3.0, 0.6, 'Склейка: [m, kd, grid, embed] = 20 чисел', '#ECEFF1', 8)
    arrow(1.3, 5.4, 2.0, 5.1)
    arrow(4.0, 7.0, 3.0, 5.1)

    # Global encoder
    box(0.8, 2.9, 3.2, 1.2, 'Global Encoder (MLP)\n20 → 256 → 256 → 256\nВектор кондиционирования:\nописывает физику задачи', '#C8E6C9', 8)
    arrow(2.5, 4.5, 2.4, 4.1)

    # Per-freq input
    box(3.5, 1.5, 4.2, 0.9, 'Для КАЖДОЙ частотной точки k:\nvход = [Re(D_hat(k)), Im(D_hat(k)), координаты, cond]\n= 18 + 3 + 256 = 277 чисел', '#FFF9C4', 8)
    arrow(2.4, 2.9, 4.0, 2.4)
    arrow(8.0, 7.0, 7.0, 2.4)

    # Freq MLP
    box(3.5, 0.1, 4.2, 1.1, 'Per-Frequency MLP\n(одна и та же MLP для ВСЕХ частот!)\n277 → 256 → 256 → 256 → 18\nM_hat(k) = I + residual;  K²: M_hat = M_hat @ M_hat', '#FFE0B2', 8)
    arrow(5.6, 1.5, 5.6, 1.2)

    # Key box
    box(7.8, 2.9, 2.0, 2.8, 'КЛЮЧЕВОЕ\n\nОдна MLP для\nвсех частот\n\nНет стенсиля →\nнет ограничения\nпо дальности\n\nВидит матрицу A\n\n228K параметров', '#FFECB3', 8)

    ax.text(0.5, 0.02, 'ShapeEncoder3D: 70K  |  Global Encoder: 104K  |  Freq MLP: 53K  |  Итого: 228K параметров',
            fontsize=9, style='italic', transform=ax.transAxes, ha='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: Training
# ═══════════════════════════════════════════════════════════════════════
def page4_training(pdf):
    fig, ax = new_page(pdf, 'Обучение')

    add_text(ax, 0.88, 'Команда запуска', fontsize=13, bold=True)
    add_code(ax, 0.84,
        'python train_v7/train.py --name spectral_v3_adda --device 0 --save \\\n'
        '    --loss adversarial --spectral --squared_kernel \\\n'
        '    --freq_hidden 256 --freq_layers 5 --global_hidden 256 --global_layers 3 \\\n'
        '    --grid_min 8 --grid_max 64 --ema_decay 0.999 --warmup_steps 500 \\\n'
        '    --num_steps 60000 --curriculum_frac 0.25 --lr 1e-3',
        fontsize=8)

    add_text(ax, 0.67, 'Параметры', fontsize=13, bold=True)

    params = [
        ['Параметр', 'Значение', 'Пояснение'],
        ['--spectral', '—', 'Включает spectral архитектуру'],
        ['--squared_kernel', '—', 'K²: M_hat = M_hat @ M_hat'],
        ['--freq_hidden', '256', 'Ширина per-frequency MLP'],
        ['--freq_layers', '5', 'Глубина per-frequency MLP'],
        ['--grid_min / max', '8 / 64', 'Диапазон grid при обучении'],
        ['--curriculum_frac', '0.25', '25% обучения: grid растёт от min до max'],
        ['--lr', '1e-3', 'Скорость обучения'],
        ['--ema_decay', '0.999', 'Скользящее среднее весов (сглаживает шум)'],
        ['--num_steps', '60000', 'Число шагов обучения (6–9 часов на RTX 3090 Ti)'],
    ]
    t = ax.table(cellText=params[1:], colLabels=params[0],
                 cellLoc='left', loc='center',
                 bbox=[0.04, 0.22, 0.92, 0.42])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    for j in range(3):
        t[0, j].set_facecolor('#1565C0')
        t[0, j].set_text_props(color='white', fontweight='bold')

    add_text(ax, 0.19, 'Что происходит на каждом шаге', fontsize=13, bold=True)
    add_text(ax, 0.15,
        '1. Генерация случайной задачи: m ∈ [1.5, 4.0], grid ∈ [8, max], случайная форма\n'
        '2. Построение FFTMatVec → получение D_hat (матрица задачи в частотной области)\n'
        '3. Нейросеть: D_hat + параметры → M_hat (прекондиционер)\n'
        '4. Adversarial probe loss: найти худший вектор z, минимизировать ||M·A·z − z||²\n'
        '5. Обратное распространение + обновление весов (AdamW + EMA)',
        fontsize=9)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5: Export & ADDA
# ═══════════════════════════════════════════════════════════════════════
def page5_export(pdf):
    fig, ax = new_page(pdf, 'Экспорт и использование в ADDA')

    add_text(ax, 0.88, 'Шаг 1: Экспорт прекондиционера', fontsize=13, bold=True)
    add_text(ax, 0.84,
        'Spectral модель предсказывает M_hat для конкретной задачи.\n'
        'Экспорт нужен для каждой комбинации (форма, grid, m, kd).',
        fontsize=10)

    add_code(ax, 0.76,
        'python apps/export_spectral_precond.py \\\n'
        '    --checkpoint results/spectral_v3_adda/best_model.pt \\\n'
        '    --shape sphere --grid 33 \\\n'
        '    --m_re 3.0 --m_im 0.0 --kd 0.4189 \\\n'
        '    --output /tmp/precond.precond')

    add_text(ax, 0.62,
        'Что делает: создаёт диполи → строит D_hat → нейросеть → M_hat → IFFT → стенсиль → файл',
        fontsize=9)

    add_text(ax, 0.56, 'Шаг 2: Запуск ADDA', fontsize=13, bold=True)

    add_code(ax, 0.51,
        'LD_LIBRARY_PATH=$HOME/.local/lib \\\n'
        'adda/src/seq/adda -grid 33 -m 3.0 0.0 -shape sphere \\\n'
        '    -iter bicgstab -eps 5 -dpl 15 \\\n'
        '    -precond /tmp/precond.precond')

    add_warning(ax, 0.38,
        '⚠  КРИТИЧЕСКИ ВАЖНО:\n\n'
        '  Используйте   -grid N -dpl 15   (точный kd = 2π/15)\n'
        '  НЕ используйте   -size X -dpl 15   (kd будет неточным!)\n\n'
        '  Малейшее расхождение kd ПОЛНОСТЬЮ УБИВАЕТ\n'
        '  прекондиционер — он ухудшает вместо ускорения!')

    add_text(ax, 0.15, 'Формат .precond файла (mode=3 CONVSAI)', fontsize=12, bold=True)
    add_text(ax, 0.11,
        'Header (40 байт): magic, n=3·N_dipoles, n_stencil, mode=3, reserved=0\n'
        'Data: стенсиль (n_stencil × 3 int32) + ядро (n_stencil × 18 float64)\n'
        'ADDA при загрузке делает FFT стенсиля → Phat. Применяет через FFT-свёртку.',
        fontsize=9)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6: ADDA changes
# ═══════════════════════════════════════════════════════════════════════
def page6_adda(pdf):
    fig, ax = new_page(pdf, 'Изменения в коде ADDA')

    files = [
        ['Файл', 'Что добавлено'],
        ['precond.h', '#define PRECOND_MODE_FFTDIRECT 4\nvoid DumpDhat(const char *filename);'],
        ['precond.c', 'DumpDhat() — экспорт D_hat в бинарный файл\n'
                      'LoadFFTDirect() — загрузка Phat напрямую (mode=4)\n'
                      'Обработка mode=4 в PrecondLoad/Apply/Free'],
        ['calculator.c', '#include "precond.h"\n'
                         'Вызов DumpDhat() после InitDmatrix()'],
        ['param.c', 'PARSE_FUNC(dump_dhat) — аргумент -dump_dhat <file>'],
        ['vars.c / vars.h', 'const char *dump_dhat_filename = NULL;'],
    ]
    t = ax.table(cellText=files[1:], colLabels=files[0],
                 cellLoc='left', loc='upper center',
                 bbox=[0.04, 0.52, 0.92, 0.35])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 2.0)
    for j in range(2):
        t[0, j].set_facecolor('#1565C0')
        t[0, j].set_text_props(color='white', fontweight='bold')

    add_text(ax, 0.48, 'Сборка', fontsize=13, bold=True)
    add_code(ax, 0.44,
        'cd adda/src\n'
        'make seq FFTW3_INC_PATH=$HOME/.local/include FFTW3_LIB_PATH=$HOME/.local/lib')

    add_text(ax, 0.34, 'Конвенция FFTMatVec (важно для воспроизведения!)', fontsize=13, bold=True)
    add_text(ax, 0.29,
        'Python FFTMatVec ДОЛЖЕН использовать ADDA-совместимую конвенцию:\n'
        '  k = 1.0,  d = kd   (так что k·d = kd — совпадает с ADDA)\n\n'
        'НЕ использовать:  k = kd,  d = 1.0\n\n'
        'Оба варианта дают одинаковый A·v, но D_hat отличается!\n'
        'Spectral модель принимает D_hat на вход — конвенция должна совпадать.',
        fontsize=9)

    add_code(ax, 0.08,
        '# В train_v7/train.py, build_fft_matvec_from_positions:\n'
        'd = kd      # ADDA convention\n'
        'k = 1.0\n'
        'return FFTMatVec(positions, k, m, d=d, device=device)')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 7: Reproduction details
# ═══════════════════════════════════════════════════════════════════════
def page7_details(pdf):
    fig, ax = new_page(pdf, 'Детали воспроизведения')

    details = [
        ('1. Identity init (нулевая инициализация)',
         'Последний слой freq MLP = нули → M_hat стартует как единичная матрица I.\n'
         'С K²: I @ I = I → на старте прекондиционер не влияет.\n'
         'Модель постепенно учится отклоняться от I.'),

        ('2. Adversarial probe loss',
         'loss = ||M·A·z − z||² / ||z||²\n'
         'z — "наихудший" вектор (10 шагов power iteration).\n'
         'Оптимизация на worst-case → M хорош для ВСЕХ направлений.'),

        ('3. LDR поляризуемость',
         'Python и ADDA используют одинаковую формулу Lattice Dispersion Relation.\n'
         'Реализована в apps/adda_matrix.py: ldr_polarizability().'),

        ('4. Размер стенсиля при экспорте',
         'M_hat → IFFT → плотное ядро (2·grid)³. Для grid=48: ~884K записей, ~135 MB.\n'
         'ADDA обрабатывает нормально (делает FFT при загрузке).'),

        ('5. Обучающие формы',
         'sphere (15%), ellipsoid (40%, случайные пропорции), cube (15%),\n'
         'cylinder (15%), capsule (15%).'),
    ]

    y = 0.88
    for title, desc in details:
        add_text(ax, y, title, fontsize=11, bold=True, color='#1565C0')
        add_text(ax, y - 0.03, desc, fontsize=9, indent=1)
        lines = desc.count('\n') + 1
        y -= 0.05 + lines * 0.025

    add_text(ax, y - 0.02, 'Ограничения', fontsize=13, bold=True)
    add_text(ax, y - 0.06,
        '• Экспорт для каждой задачи (shape + grid + m + kd) — не универсальный файл\n'
        '• kd должен ТОЧНО совпадать с 2π/dpl.  Использовать -grid N, НЕ -size X\n'
        '• Не тестировалось на m_im > 0 и grid > 64\n'
        '• Большие файлы стенсиля для grid ≥ 48',
        fontsize=9)

    add_text(ax, y - 0.22, 'Быстрый старт', fontsize=13, bold=True)
    add_code(ax, y - 0.27,
        '# 1. Обучение (6-9 часов)\n'
        'python train_v7/train.py --name my_spec --device 0 --save \\\n'
        '  --loss adversarial --spectral --squared_kernel \\\n'
        '  --freq_hidden 256 --freq_layers 5 --grid_min 8 --grid_max 64 \\\n'
        '  --ema_decay 0.999 --num_steps 60000 --lr 1e-3\n\n'
        '# 2. Экспорт\n'
        'python apps/export_spectral_precond.py \\\n'
        '  --checkpoint results/my_spec/best_model.pt \\\n'
        '  --shape sphere --grid 33 --m_re 3.0 --kd 0.4189 --output /tmp/p.precond\n\n'
        '# 3. ADDA\n'
        'adda -grid 33 -m 3.0 0.0 -shape sphere -eps 5 -dpl 15 -precond /tmp/p.precond',
        fontsize=7.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 8: Heatmap
# ═══════════════════════════════════════════════════════════════════════
def page8_heatmap(pdf):
    data_file = 'results/spectral_adda_heatmap/data.jsonl'
    if not os.path.exists(data_file):
        fig, ax = new_page(pdf, 'Результаты (heatmap)')
        add_text(ax, 0.5, 'Heatmap данные ещё не готовы', fontsize=16)
        pdf.savefig(fig); plt.close(); return

    data = [json.loads(l) for l in open(data_file)]
    if len(data) < 4:
        fig, ax = new_page(pdf, 'Результаты (heatmap)')
        add_text(ax, 0.5, f'Только {len(data)} результатов, нужно больше', fontsize=16)
        pdf.savefig(fig); plt.close(); return

    shapes = sorted(set(d["shape"] for d in data))
    grids = sorted(set(d["grid"] for d in data))
    m_vals = sorted(set(d["m_re"] for d in data))

    n = len(shapes)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)
    fig.suptitle(f'ADDA: Spectral (228K) vs K² v3 (13.7M) — ускорение по итерациям [{len(data)} случаев]',
                 fontsize=14, fontweight='bold', y=1.01)

    for si, shape in enumerate(shapes):
        for col, (key, title) in enumerate([
            ("iter_spd_spectral", f"{shape} — Spectral"),
            ("iter_spd_k2v3", f"{shape} — K² v3"),
        ]):
            ax = axes[si, col]
            arr = np.full((len(m_vals), len(grids)), np.nan)
            for d in data:
                if d["shape"] != shape or key not in d: continue
                if d["grid"] in grids and d["m_re"] in m_vals:
                    arr[m_vals.index(d["m_re"]), grids.index(d["grid"])] = d[key]

            vmax = max(50, np.nanmax(arr) if np.any(~np.isnan(arr)) else 50)
            im = ax.imshow(arr, aspect='auto', origin='lower',
                          cmap='RdYlGn', vmin=0, vmax=vmax)
            ax.set_xticks(range(len(grids)))
            ax.set_xticklabels(grids, fontsize=10)
            ax.set_yticks(range(len(m_vals)))
            ax.set_yticklabels(m_vals, fontsize=10)
            ax.set_xlabel("Grid", fontsize=11)
            ax.set_ylabel("m_re", fontsize=11)
            ax.set_title(title, fontsize=11, fontweight='bold')
            for mi in range(len(m_vals)):
                for gi in range(len(grids)):
                    v = arr[mi, gi]
                    if np.isnan(v): continue
                    color = 'white' if v < 5 else 'black'
                    ax.text(gi, mi, f"{v:.1f}", ha='center', va='center',
                           fontsize=9, color=color, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Ускорение (итерации)', fraction=0.046)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 9: Full results table
# ═══════════════════════════════════════════════════════════════════════
def page9_table(pdf):
    data_file = 'results/spectral_adda_heatmap/data.jsonl'
    if not os.path.exists(data_file): return

    data = [json.loads(l) for l in open(data_file)]
    data.sort(key=lambda x: (x['shape'], x['grid'], x['m_re']))

    header = ['Конфигурация', 'Без прекон.', 'Spectral', 'Ускорение', 'K² v3', 'Ускорение']
    rows = []
    for d in data:
        tag = f"{d['shape']} g{d['grid']} m{d['m_re']}"
        s = d.get('iter_spd_spectral', '—')
        k = d.get('iter_spd_k2v3', '—')
        rows.append([tag, str(d.get('iter_base', '?')),
                     str(d.get('iter_spectral', '?')),
                     f"{s}x" if isinstance(s, (int, float)) else s,
                     str(d.get('iter_k2v3', '?')),
                     f"{k}x" if isinstance(k, (int, float)) else k])

    chunk = 30
    for i in range(0, len(rows), chunk):
        fig, ax = new_page(pdf, 'Полная таблица результатов ADDA')
        batch = rows[i:i+chunk]

        t = ax.table(cellText=batch, colLabels=header,
                    cellLoc='center', loc='upper center',
                    bbox=[0.02, 0.02, 0.96, 0.88])
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.3)

        for j in range(6):
            t[0, j].set_facecolor('#1565C0')
            t[0, j].set_text_props(color='white', fontweight='bold', fontsize=8)

        for ri, row in enumerate(batch, 1):
            try:
                sv = float(row[3].rstrip('x'))
                kv = float(row[5].rstrip('x'))
                if sv > kv:
                    t[ri, 2].set_facecolor('#E8F5E9')
                    t[ri, 3].set_facecolor('#E8F5E9')
            except: pass

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def main():
    output = os.path.join(os.path.dirname(__file__), 'spectral_guide.pdf')
    with PdfPages(output) as pdf:
        page1_intro(pdf)
        page2_glossary(pdf)
        page3_architecture(pdf)
        page4_training(pdf)
        page5_export(pdf)
        page6_adda(pdf)
        page7_details(pdf)
        page8_heatmap(pdf)
        page9_table(pdf)
    print(f"PDF saved: {output}")


if __name__ == '__main__':
    main()
