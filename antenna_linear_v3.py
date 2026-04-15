#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Программа для решения прямой и обратной задач линейной антенны.

По плану, приведённому ранее, в этой программе реализованы две основные задачи:

1. **Прямая задача** — расчёт диаграммы направленности (ДН) по заданному
   амплитудно‑фазовому распределению поля на линейной апертуре антенны.
   Пользователь задаёт геометрические параметры (длину антенны, длину волны,
   число дискретных точек), выбирает тип амплитудного и фазового распределения,
   после чего программа вычисляет комплексное распределение на апертуре,
   строит соответствующую ДН и выводит основные характеристики (ширина
   главного лепестка, уровень первого бокового лепестка, направление
   максимума).

2. **Обратная задача** — синтез амплитудно‑фазового распределения по заданной
   требуемой диаграмме направленности. В настоящей версии используется
   упрощённый метод — решение задачи наименьших квадратов для восстановления
   комплексного распределения на апертуре по известному амплитудному
   распределению ДН (с нулевой фазой). Это позволяет наглядно показать
   идею обратной задачи, хотя в реальных задачах синтез требует более
   сложных методов.

Графический интерфейс построен на Tkinter, а для построения графиков
используется matplotlib. Код разбит на компактные функции для удобства
понимания и возможного расширения.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib

# Используем backend, совместимый с Tkinter, при наличии.
# В средах без графического интерфейса (например, при тестах) используем 'Agg'.
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    matplotlib.use("Agg")
    # В headless‑режиме холст Tk недоступен
    FigureCanvasTkAgg = None
from matplotlib.figure import Figure
import csv


def safe_eval(expression: str, local_vars: dict):
    """Безопасно вычисляет пользовательское выражение.

    Используется для интерпретации аналитических формул, вводимых
    пользователем. Разрешены только функции numpy и некоторые
    базовые математические функции. В выражении можно использовать
    элементы из local_vars (например, массив x, theta, z). Если
    выражение некорректно, возбуждается исключение.

    Args:
        expression: строка с выражением, например "np.sin(x)".
        local_vars: словарь локальных переменных, доступных в выражении.

    Returns:
        Результат вычисления выражения.
    """
    # Разрешённые имена
    # Разрешённые имена (функции и константы)
    allowed_names = {
        'sin': np.sin,
        'cos': np.cos,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': np.pi,
    }
    # Дополняем локальными переменными (например x, L, theta, z)
    if local_vars:
        allowed_names.update(local_vars)
    # Запрещаем доступ к builtins и другим объектам
    try:
        return eval(expression, {"__builtins__": None}, allowed_names)
    except Exception as exc:
        raise ValueError(f"Невозможно вычислить выражение '{expression}': {exc}")


def amplitude_distribution(x: np.ndarray, dist_type: str) -> np.ndarray:
    """Возвращает амплитудное распределение для линейной апертуры.

    Args:
        x: массив координат по апертуре (в метрах).
        dist_type: тип распределения ('uniform', 'cosine', 'gaussian').

    Returns:
        amplitude: амплитуда в каждой точке x.
    """
    if dist_type == 'uniform':
        return np.ones_like(x)
    elif dist_type == 'cosine':
        # Косинусное распределение: cos(pi * x / L)
        # x в диапазоне [-L/2, L/2], поэтому нормируем на длину массива
        return np.cos(np.pi * x / (x[-1] - x[0]))
    elif dist_type == 'gaussian':
        # Гауссово распределение: exp(- (2*x/L)^2)
        L = x[-1] - x[0]
        return np.exp(-((2 * x / L) ** 2))
    else:
        # По умолчанию равномерное
        return np.ones_like(x)


def phase_distribution(x: np.ndarray, dist_type: str, param: float) -> np.ndarray:
    """Возвращает фазовое распределение для линейной апертуры.

    Args:
        x: массив координат по апертуре (в метрах).
        dist_type: тип фазового распределения ('zero', 'linear').
        param: параметр фазового распределения (угловой коэффициент для
            линейного наклона). В радианах на метр.

    Returns:
        phase: фаза (в радианах) в каждой точке x.
    """
    if dist_type == 'zero':
        return np.zeros_like(x)
    elif dist_type == 'linear':
        # Линейный наклон фазы: phi(x) = param * x
        return param * x
    else:
        return np.zeros_like(x)


def compute_direct(L: float, wavelength: float, N: int,
                   amp_type: str, phase_type: str, phase_param: float,
                   amp_expr: str = None, phase_expr: str = None,
                   M: int = 361):
    """Решает прямую задачу для линейной антенны.

    Args:
        L: длина антенны (метры).
        wavelength: длина волны (метры).
        N: число дискретных точек по апертуре.
        amp_type: тип амплитудного распределения.
        phase_type: тип фазового распределения.
        phase_param: параметр фазового распределения (для линейного наклона).

    Returns:
        x: координаты по апертуре (массив длины N).
        amp: амплитуда на апертуре (N).
        phase: фаза на апертуре (N).
        theta_deg: массив углов в градусах (-90° … 90°).
        pattern_lin: нормированная ДН в линейном масштабе (размер 361).
        pattern_db: ДН в дБ (размер 361).
        metrics: словарь характеристик ('beamwidth_deg', 'sidelobe_level_db', 'max_direction_deg').
    """
    # Коэффициент волнового числа
    k = 2 * np.pi / wavelength
    # Координаты по апертуре
    x = np.linspace(-L / 2.0, L / 2.0, N)
    # Амплитуда и фаза
    amp = amplitude_distribution(x, amp_type)
    phase = phase_distribution(x, phase_type, phase_param)
    # Если задано пользовательское выражение для амплитуды, вычисляем его
    if amp_expr:
        try:
            # Локальные переменные для выражения: x, L, np, pi
            local_vars = {
                'x': x,
                'L': L,
            }
            amp_eval = safe_eval(amp_expr, local_vars)
            # Проверяем размерность
            if np.asarray(amp_eval).shape != x.shape:
                raise ValueError("Формула амплитуды должна возвращать массив той же длины, что и x")
            # Берём действительную часть
            amp = np.real(np.asarray(amp_eval))
        except Exception as e:
            raise ValueError(f"Ошибка в формуле амплитуды: {e}")
    # Аналогично для фазы
    if phase_expr:
        try:
            local_vars = {
                'x': x,
                'L': L,
            }
            phase_eval = safe_eval(phase_expr, local_vars)
            if np.asarray(phase_eval).shape != x.shape:
                raise ValueError("Формула фазы должна возвращать массив той же длины, что и x")
            phase = np.asarray(phase_eval)
        except Exception as e:
            raise ValueError(f"Ошибка в формуле фазы: {e}")
    # Комплексное распределение
    F = amp * np.exp(1j * phase)
    # Углы для вычисления ДН
    theta_deg = np.linspace(-90.0, 90.0, M)
    theta_rad = np.deg2rad(theta_deg)
    # Вычисляем интеграл как дискретную сумму
    pattern_complex = np.zeros_like(theta_rad, dtype=complex)
    # Шаг апертуры (L/N) — используется для приближения интеграла
    dx = L / (N - 1)
    # Матрица фазовых множителей
    # Чтобы ускорить вычисления, можно сразу сформировать матрицу exp(j*k*x*sin(theta))
    # Размер (361, N)
    sin_theta = np.sin(theta_rad).reshape(-1, 1)  # (M, 1)
    exp_matrix = np.exp(1j * k * x.reshape(1, -1) * sin_theta)  # (M, N)
    pattern_complex = (exp_matrix @ F) * dx
    # Амплитуда
    pattern_mag = np.abs(pattern_complex)
    # Нормировка
    pattern_mag_norm = pattern_mag / np.max(pattern_mag)
    # В дБ
    pattern_db = 20 * np.log10(pattern_mag_norm + 1e-12)
    # Характеристики: ширина главного лепестка, уровень боковых лепестков и направление максимума
    metrics = {}
    # Направление максимума
    max_index = np.argmax(pattern_mag_norm)
    max_direction_deg = theta_deg[max_index]
    metrics['max_direction_deg'] = max_direction_deg
    # Ширина главного лепестка на уровне -3 дБ
    # Находим ближайшие точки, где ДН падает до -3 дБ от максимума
    half_power_level = -3.0
    # Ищем точки слева и справа от максимума
    # Индекс максимума
    # Слева
    left_indices = np.where(pattern_db[:max_index] <= half_power_level)[0]
    right_indices = np.where(pattern_db[max_index + 1:] <= half_power_level)[0]
    beamwidth_deg = None
    if len(left_indices) > 0 and len(right_indices) > 0:
        left_cross = theta_deg[left_indices[-1]]
        right_cross = theta_deg[max_index + 1 + right_indices[0]]
        beamwidth_deg = right_cross - left_cross
    metrics['beamwidth_deg'] = beamwidth_deg
    # Уровень первого бокового лепестка
    # Ищем все локальные максимумы, кроме главного
    sidelobe_level_db = None
    try:
        import scipy.signal  # noqa
        # Используем поиск пиков
    except Exception:
        scipy_installed = False
    else:
        scipy_installed = True
    if scipy_installed:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(pattern_mag_norm)
        # Удаляем главный максимум
        if max_index in peaks:
            peaks = peaks[peaks != max_index]
        # Считаем высоту пиков
        if len(peaks) > 0:
            peak_values_db = pattern_db[peaks]
            sidelobe_level_db = np.max(peak_values_db)
    else:
        # Если SciPy недоступен, ищем просто второй максимум по уровню
        temp = pattern_mag_norm.copy()
        temp[max_index] = 0.0
        second_max_index = np.argmax(temp)
        sidelobe_level_db = pattern_db[second_max_index]
    metrics['sidelobe_level_db'] = sidelobe_level_db
    return x, amp, phase, theta_deg, pattern_mag_norm, pattern_db, metrics


def compute_inverse(L: float, wavelength: float, N: int,
                    pattern_type: str, param: float,
                    pattern_expr: str = None,
                    M: int = 361):
    """Решает упрощённую обратную задачу для линейной антенны.

    Восстанавливает комплексное распределение на апертуре по заданной
    амплитудной диаграмме направленности с нулевой фазой. Реализовано
    посредством метода наименьших квадратов.

    Args:
        L: длина антенны (м).
        wavelength: длина волны (м).
        N: число точек апертуры.
        pattern_type: тип требуемой ДН ('cosine_squared', 'gaussian').
        param: параметр требуемой ДН (степень для cosine_squared,
            ширина для gaussian).

    Returns:
        x: координаты апертуры (N).
        amp: амплитуда восстановленного АФР (N).
        phase: фаза восстановленного АФР (N).
        theta_deg: углы (361).
        pattern_target: нормированная требуемая ДН (361).
        pattern_synth: нормированная синтезированная ДН (361).
        metrics: словарь с ошибкой синтеза (RMS).
    """
    k = 2 * np.pi / wavelength
    x = np.linspace(-L / 2.0, L / 2.0, N)
    theta_deg = np.linspace(-90.0, 90.0, M)
    theta_rad = np.deg2rad(theta_deg)
    # Формируем требуемую амплитуду
    if pattern_type == 'cosine_squared':
        # |cos(theta)|^n
        pattern_target = np.abs(np.cos(theta_rad)) ** param
    elif pattern_type == 'gaussian':
        # exp(-(theta/b)^2)
        # param — ширина в радианах (например 0.4 для ширины ~45°)
        pattern_target = np.exp(-(theta_rad / param) ** 2)
    elif pattern_type == 'custom_formula' and pattern_expr:
        try:
            # Определяем переменные, доступные пользователю
            # theta — угол в радианах, theta_deg — в градусах
            # z = k*L*sin(theta)
            local_vars = {
                'theta': theta_rad,
                'theta_deg': theta_deg,
                'z': k * L * np.sin(theta_rad),
                'L': L,
                'k': k,
            }
            pattern_eval = safe_eval(pattern_expr, local_vars)
            # Проверяем размерность
            if np.asarray(pattern_eval).shape != theta_rad.shape:
                raise ValueError("Формула ДН должна возвращать массив той же длины, что и theta")
            pattern_target = np.asarray(pattern_eval)
            # Берём модуль, если комплексное
            pattern_target = np.abs(pattern_target)
        except Exception as e:
            raise ValueError(f"Ошибка в формуле ДН: {e}")
    else:
        # по умолчанию cos^2
        pattern_target = np.abs(np.cos(theta_rad)) ** param
    # Нормировка
    if np.max(pattern_target) > 0:
        pattern_target = pattern_target / np.max(pattern_target)
    # Требуемое комплексное поле (фаза = 0)
    D = pattern_target.astype(complex)
    # Формируем матрицу A размера (361, N): A[i, j] = exp(j*k*x_j*sin(theta_i)) * dx
    dx = L / (N - 1)
    sin_theta = np.sin(theta_rad).reshape(-1, 1)
    A = np.exp(1j * k * x.reshape(1, -1) * sin_theta) * dx
    # Решаем систему A * F = D в смысле наименьших квадратов
    # F — искомое комплексное распределение (N)
    # Для стабильности используем псевдообратную матрицу
    # В случае плохо обусловленных задач это позволит получить приближённое решение
    try:
        # Находим наименьших квадратов решение
        # numpy.linalg.lstsq возвращает минимизирующее ||A F - D||
        F, residuals, rank, s = np.linalg.lstsq(A, D, rcond=None)
    except Exception:
        # В случае ошибки возвращаем нули
        F = np.zeros_like(x, dtype=complex)
    # Амплитуда и фаза
    amp = np.abs(F)
    phase = np.angle(F)
    # Синтезированная ДН
    pattern_synth_complex = A @ F
    pattern_synth = np.abs(pattern_synth_complex)
    if np.max(pattern_synth) > 0:
        pattern_synth /= np.max(pattern_synth)
    # Ошибка синтеза (среднеквадратичная)
    error_rms = np.sqrt(np.mean((pattern_synth - pattern_target) ** 2))
    metrics = {'synthesis_rms_error': error_rms}
    return x, amp, phase, theta_deg, pattern_target, pattern_synth, metrics


class AntennaApp(tk.Tk):
    """Основной класс приложения."""
    def __init__(self):
        super().__init__()
        self.title("Прямая и обратная задачи линейной антенны")
        self.geometry("900x700")
        # Настраиваем notebook (вкладки)
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        # Вкладка для прямой задачи
        self.direct_frame = ttk.Frame(notebook)
        notebook.add(self.direct_frame, text="Прямая задача")
        # Вкладка для обратной задачи
        self.inverse_frame = ttk.Frame(notebook)
        notebook.add(self.inverse_frame, text="Обратная задача")
        # Создаём содержимое
        self.create_direct_tab()
        self.create_inverse_tab()

    def create_direct_tab(self):
        # Параметры ввода для прямой задачи
        frame = self.direct_frame
        # Верхняя часть — поля ввода
        input_frame = ttk.Frame(frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        # Длина антенны
        ttk.Label(input_frame, text="Длина антенны L (м):").grid(row=0, column=0, sticky=tk.W)
        self.direct_L_var = tk.DoubleVar(value=1.0)
        ttk.Entry(input_frame, textvariable=self.direct_L_var, width=10).grid(row=0, column=1, padx=5)
        # Длина волны
        ttk.Label(input_frame, text="Длина волны λ (м):").grid(row=1, column=0, sticky=tk.W)
        self.direct_lambda_var = tk.DoubleVar(value=0.1)
        ttk.Entry(input_frame, textvariable=self.direct_lambda_var, width=10).grid(row=1, column=1, padx=5)
        # Число точек
        ttk.Label(input_frame, text="Число точек N:").grid(row=2, column=0, sticky=tk.W)
        self.direct_N_var = tk.IntVar(value=101)
        ttk.Entry(input_frame, textvariable=self.direct_N_var, width=10).grid(row=2, column=1, padx=5)
        # Амплитудное распределение
        ttk.Label(input_frame, text="Амплитуда:").grid(row=0, column=2, sticky=tk.W)
        self.direct_amp_var = tk.StringVar(value='uniform')
        amp_options = {'Равномерное': 'uniform', 'Косинусное': 'cosine', 'Гауссово': 'gaussian', 'Произвольная': 'custom'}
        amp_menu = ttk.OptionMenu(input_frame, self.direct_amp_var, 'uniform', *amp_options.values())
        amp_menu.grid(row=0, column=3, padx=5)
        # Фазовое распределение
        ttk.Label(input_frame, text="Фаза:").grid(row=1, column=2, sticky=tk.W)
        self.direct_phase_type_var = tk.StringVar(value='zero')
        phase_options = {'Нулевая': 'zero', 'Линейная': 'linear', 'Произвольная': 'custom'}
        phase_menu = ttk.OptionMenu(input_frame, self.direct_phase_type_var, 'zero', *phase_options.values())
        phase_menu.grid(row=1, column=3, padx=5)
        # Параметр фазы
        ttk.Label(input_frame, text="Параметр фазы (рад/м):").grid(row=2, column=2, sticky=tk.W)
        self.direct_phase_param_var = tk.DoubleVar(value=0.0)
        ttk.Entry(input_frame, textvariable=self.direct_phase_param_var, width=10).grid(row=2, column=3, padx=5)

        # Пользовательская формула амплитуды
        ttk.Label(input_frame, text="Формула амплитуды f(x):").grid(row=4, column=0, sticky=tk.W)
        self.direct_amp_expr_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.direct_amp_expr_var, width=30).grid(row=4, column=1, columnspan=3, padx=5, pady=2, sticky=tk.W)
        # Пользовательская формула фазы
        ttk.Label(input_frame, text="Формула фазы φ(x):").grid(row=5, column=0, sticky=tk.W)
        self.direct_phase_expr_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.direct_phase_expr_var, width=30).grid(row=5, column=1, columnspan=3, padx=5, pady=2, sticky=tk.W)

        # Число углов для расчёта ДН
        ttk.Label(input_frame, text="Число углов M:").grid(row=6, column=0, sticky=tk.W)
        self.direct_M_var = tk.IntVar(value=361)
        ttk.Entry(input_frame, textvariable=self.direct_M_var, width=10).grid(row=6, column=1, padx=5)

        # Подсказка по формуле амплитуды и фазы
        ttk.Label(input_frame, text="Доступные переменные: x, L; функции: sin, cos, exp, sqrt, abs, pi.").grid(row=7, column=0, columnspan=4, sticky=tk.W, pady=(5,0))
        # Кнопка расчёта
        ttk.Button(input_frame, text="Рассчитать", command=self.on_calculate_direct).grid(row=3, column=0, columnspan=4, pady=5)
        # Графики
        self.direct_fig = Figure(figsize=(6, 4))
        self.direct_ax1 = self.direct_fig.add_subplot(211)
        self.direct_ax2 = self.direct_fig.add_subplot(212)
        self.direct_canvas = FigureCanvasTkAgg(self.direct_fig, master=frame)
        self.direct_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Метрики
        self.direct_metrics_text = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.direct_metrics_text, justify=tk.LEFT, foreground='blue').pack(pady=5)

        # Текстовая область для вывода расчётных данных
        self.direct_output_text = ScrolledText(frame, height=6)
        self.direct_output_text.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Кнопка сохранения результатов
        ttk.Button(frame, text="Сохранить результаты", command=self.save_direct_results).pack(pady=(0, 10))

    def create_inverse_tab(self):
        frame = self.inverse_frame
        input_frame = ttk.Frame(frame)
        input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        # Длина антенны
        ttk.Label(input_frame, text="Длина антенны L (м):").grid(row=0, column=0, sticky=tk.W)
        self.inv_L_var = tk.DoubleVar(value=1.0)
        ttk.Entry(input_frame, textvariable=self.inv_L_var, width=10).grid(row=0, column=1, padx=5)
        # Длина волны
        ttk.Label(input_frame, text="Длина волны λ (м):").grid(row=1, column=0, sticky=tk.W)
        self.inv_lambda_var = tk.DoubleVar(value=0.1)
        ttk.Entry(input_frame, textvariable=self.inv_lambda_var, width=10).grid(row=1, column=1, padx=5)
        # Число точек
        ttk.Label(input_frame, text="Число точек N:").grid(row=2, column=0, sticky=tk.W)
        self.inv_N_var = tk.IntVar(value=51)
        ttk.Entry(input_frame, textvariable=self.inv_N_var, width=10).grid(row=2, column=1, padx=5)

        # Число углов для ДН
        ttk.Label(input_frame, text="Число углов M:").grid(row=3, column=0, sticky=tk.W)
        self.inv_M_var = tk.IntVar(value=361)
        ttk.Entry(input_frame, textvariable=self.inv_M_var, width=10).grid(row=3, column=1, padx=5)
        # Тип целевой ДН
        ttk.Label(input_frame, text="Тип требуемой ДН:").grid(row=0, column=2, sticky=tk.W)
        self.inv_pattern_type_var = tk.StringVar(value='cosine_squared')
        pattern_options = {'|cos(theta)|^n': 'cosine_squared', 'Gaussian': 'gaussian', 'Формула': 'custom_formula'}
        pattern_menu = ttk.OptionMenu(input_frame, self.inv_pattern_type_var, 'cosine_squared', *pattern_options.values())
        pattern_menu.grid(row=0, column=3, padx=5)
        # Параметр ДН
        ttk.Label(input_frame, text="Параметр (n или ширина):").grid(row=1, column=2, sticky=tk.W)
        self.inv_param_var = tk.DoubleVar(value=2.0)
        ttk.Entry(input_frame, textvariable=self.inv_param_var, width=10).grid(row=1, column=3, padx=5)

        # Поле ввода пользовательской формулы ДН (если выбран вариант 'Формула')
        ttk.Label(input_frame, text="Формула ДН D(θ или z):").grid(row=2, column=2, sticky=tk.W)
        self.inv_pattern_expr_var = tk.StringVar(value="")
        ttk.Entry(input_frame, textvariable=self.inv_pattern_expr_var, width=30).grid(row=2, column=3, padx=5, pady=2, sticky=tk.W)

        # Подсказка по формуле ДН
        ttk.Label(input_frame, text="Переменные: theta (рад), theta_deg (град), z=k*L*sin(theta); функции: sin, cos, exp, sqrt, abs, pi.").grid(row=3, column=2, columnspan=2, sticky=tk.W, pady=(5,0))
        # Кнопка синтеза
        ttk.Button(input_frame, text="Синтезировать", command=self.on_calculate_inverse).grid(row=4, column=0, columnspan=4, pady=5)
        # Графики
        self.inverse_fig = Figure(figsize=(6, 4))
        self.inv_ax1 = self.inverse_fig.add_subplot(211)
        self.inv_ax2 = self.inverse_fig.add_subplot(212)
        self.inverse_canvas = FigureCanvasTkAgg(self.inverse_fig, master=frame)
        self.inverse_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Метрики
        self.inverse_metrics_text = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.inverse_metrics_text, justify=tk.LEFT, foreground='blue').pack(pady=5)

        # Текстовая область для вывода расчётных данных
        self.inverse_output_text = ScrolledText(frame, height=6)
        self.inverse_output_text.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Кнопка сохранения результатов
        ttk.Button(frame, text="Сохранить результаты", command=self.save_inverse_results).pack(pady=(0, 10))

    def on_calculate_direct(self):
        """Обработчик кнопки расчёта прямой задачи."""
        try:
            L = float(self.direct_L_var.get())
            lam = float(self.direct_lambda_var.get())
            N = int(self.direct_N_var.get())
            amp_type = self.direct_amp_var.get()
            phase_type = self.direct_phase_type_var.get()
            phase_param = float(self.direct_phase_param_var.get())
            amp_expr = self.direct_amp_expr_var.get().strip()
            phase_expr = self.direct_phase_expr_var.get().strip()
            M = int(self.direct_M_var.get())
            if N < 2:
                raise ValueError("Число точек должно быть не менее 2")
            if M < 3:
                raise ValueError("Число углов должно быть не менее 3")
        except Exception as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        # Вычисления
        try:
            x, amp, phase, theta_deg, pattern_lin, pattern_db, metrics = compute_direct(
                L, lam, N, amp_type, phase_type, phase_param,
                amp_expr=amp_expr if amp_expr else None,
                phase_expr=phase_expr if phase_expr else None,
                M=M)
        except Exception as e:
            messagebox.showerror("Ошибка вычислений", str(e))
            return
        # Строим графики
        self.direct_ax1.clear()
        self.direct_ax2.clear()
        # График АФР амплитуды и фазы
        self.direct_ax1.set_title("Амплитуда и фаза на апертуре")
        self.direct_ax1.plot(x, amp, label="Амплитуда")
        self.direct_ax1.set_ylabel("Амплитуда")
        self.direct_ax1_twin = self.direct_ax1.twinx()
        self.direct_ax1_twin.plot(x, phase, color='red', linestyle='--', label="Фаза")
        self.direct_ax1_twin.set_ylabel("Фаза (рад)")
        self.direct_ax1.set_xlabel("Координата x (м)")
        # График ДН в дБ
        self.direct_ax2.set_title("Диаграмма направленности")
        self.direct_ax2.plot(theta_deg, pattern_db, label="ДН (дБ)")
        self.direct_ax2.set_ylabel("Уровень (дБ)")
        self.direct_ax2.set_xlabel("Угол θ (град)")
        self.direct_ax2.grid(True)
        # Отображаем
        self.direct_fig.tight_layout()
        self.direct_canvas.draw()
        # Метрики
        metrics_lines = []
        metrics_lines.append(f"Максимум при θ = {metrics['max_direction_deg']:.1f}°")
        if metrics['beamwidth_deg'] is not None:
            metrics_lines.append(f"Ширина главного лепестка (на уровне -3 дБ): {metrics['beamwidth_deg']:.2f}°")
        if metrics['sidelobe_level_db'] is not None:
            metrics_lines.append(f"Уровень первого бокового лепестка: {metrics['sidelobe_level_db']:.2f} дБ")
        self.direct_metrics_text.set("\n".join(metrics_lines))
        # Заполняем текстовую область результатами
        self.direct_output_text.delete('1.0', tk.END)
        # Выводим первые 10 значений амплитуды и фазы
        self.direct_output_text.insert(tk.END, "Первые 10 значений амплитуды: \n")
        self.direct_output_text.insert(tk.END, np.array2string(amp[:10], precision=4, separator=', ') + "\n")
        self.direct_output_text.insert(tk.END, "Первые 10 значений фазы (рад): \n")
        self.direct_output_text.insert(tk.END, np.array2string(phase[:10], precision=4, separator=', ') + "\n")
        self.direct_output_text.insert(tk.END, "\n")

        # Сохраняем последние результаты для возможности экспорта
        self.direct_last_results = {
            'x': x,
            'amp': amp,
            'phase': phase,
            'theta': theta_deg,
            'pattern_db': pattern_db,
            'pattern_lin': pattern_lin,
        }

    def on_calculate_inverse(self):
        """Обработчик кнопки синтеза обратной задачи."""
        try:
            L = float(self.inv_L_var.get())
            lam = float(self.inv_lambda_var.get())
            N = int(self.inv_N_var.get())
            pattern_type = self.inv_pattern_type_var.get()
            param = float(self.inv_param_var.get())
            pattern_expr = self.inv_pattern_expr_var.get().strip()
            M = int(self.inv_M_var.get())
            if N < 2:
                raise ValueError("Число точек должно быть не менее 2")
            if M < 3:
                raise ValueError("Число углов должно быть не менее 3")
        except Exception as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        # Вычисляем
        try:
            x, amp, phase, theta_deg, pattern_target, pattern_synth, metrics = compute_inverse(
                L, lam, N, pattern_type, param,
                pattern_expr=pattern_expr if pattern_type == 'custom_formula' and pattern_expr else None,
                M=M)
        except Exception as e:
            messagebox.showerror("Ошибка вычислений", str(e))
            return
        # Графики
        self.inv_ax1.clear()
        self.inv_ax2.clear()
        # АФР
        self.inv_ax1.set_title("Амплитуда и фаза синтезированного АФР")
        self.inv_ax1.plot(x, amp, label="Амплитуда")
        self.inv_ax1.set_ylabel("Амплитуда")
        self.inv_ax1_twin = self.inv_ax1.twinx()
        self.inv_ax1_twin.plot(x, phase, color='red', linestyle='--', label="Фаза")
        self.inv_ax1_twin.set_ylabel("Фаза (рад)")
        self.inv_ax1.set_xlabel("Координата x (м)")
        # ДН: требуемая и синтезированная
        self.inv_ax2.set_title("Требуемая и синтезированная ДН")
        self.inv_ax2.plot(theta_deg, pattern_target, label="Требуемая ДН")
        self.inv_ax2.plot(theta_deg, pattern_synth, linestyle='--', label="Синтезированная ДН")
        self.inv_ax2.set_ylabel("Нормированный уровень")
        self.inv_ax2.set_xlabel("Угол θ (град)")
        self.inv_ax2.grid(True)
        self.inv_ax2.legend()
        self.inverse_fig.tight_layout()
        self.inverse_canvas.draw()
        # Метрики
        self.inverse_metrics_text.set(
            f"Среднеквадратичная ошибка синтеза: {metrics['synthesis_rms_error']:.4f}")

        # Заполняем текстовую область результатами
        self.inverse_output_text.delete('1.0', tk.END)
        # Выводим первые 10 значений амплитуды и фазы
        self.inverse_output_text.insert(tk.END, "Первые 10 значений амплитуды: \n")
        self.inverse_output_text.insert(tk.END, np.array2string(amp[:10], precision=4, separator=', ') + "\n")
        self.inverse_output_text.insert(tk.END, "Первые 10 значений фазы (рад): \n")
        self.inverse_output_text.insert(tk.END, np.array2string(phase[:10], precision=4, separator=', ') + "\n")
        self.inverse_output_text.insert(tk.END, "\n")

        # Сохраняем последние результаты для возможности экспорта
        self.inverse_last_results = {
            'x': x,
            'amp': amp,
            'phase': phase,
            'theta': theta_deg,
            'pattern_target': pattern_target,
            'pattern_synth': pattern_synth,
        }

    def save_direct_results(self):
        """Сохранить результаты прямой задачи в CSV-файл."""
        if not hasattr(self, 'direct_last_results'):
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт прямой задачи.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                 filetypes=[('CSV файлы', '*.csv'), ('Все файлы', '*.*')],
                                                 title="Сохранить результаты")
        if not file_path:
            return
        results = self.direct_last_results
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Сохраняем амплитудно‑фазовое распределение
                writer.writerow(['# Амплитудно-фазовое распределение'])
                writer.writerow(['x', 'amp', 'phase'])
                for x_i, amp_i, phase_i in zip(results['x'], results['amp'], results['phase']):
                    writer.writerow([x_i, amp_i, phase_i])
                writer.writerow([])
                # Сохраняем диаграмму направленности
                writer.writerow(['# Диаграмма направленности'])
                writer.writerow(['theta (deg)', 'pattern_db', 'pattern_lin'])
                for th_i, db_i, lin_i in zip(results['theta'], results['pattern_db'], results['pattern_lin']):
                    writer.writerow([th_i, db_i, lin_i])
            messagebox.showinfo("Сохранено", f"Результаты сохранены в файл {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_inverse_results(self):
        """Сохранить результаты обратной задачи в CSV-файл."""
        if not hasattr(self, 'inverse_last_results'):
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт обратной задачи.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.csv',
                                                 filetypes=[('CSV файлы', '*.csv'), ('Все файлы', '*.*')],
                                                 title="Сохранить результаты")
        if not file_path:
            return
        results = self.inverse_last_results
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Сохраняем амплитудно‑фазовое распределение
                writer.writerow(['# Амплитудно-фазовое распределение'])
                writer.writerow(['x', 'amp', 'phase'])
                for x_i, amp_i, phase_i in zip(results['x'], results['amp'], results['phase']):
                    writer.writerow([x_i, amp_i, phase_i])
                writer.writerow([])
                # Сохраняем требуемую и синтезированную ДН
                writer.writerow(['# Требуемая и синтезированная ДН'])
                writer.writerow(['theta (deg)', 'pattern_target', 'pattern_synth'])
                for th_i, target_i, synth_i in zip(results['theta'], results['pattern_target'], results['pattern_synth']):
                    writer.writerow([th_i, target_i, synth_i])
            messagebox.showinfo("Сохранено", f"Результаты сохранены в файл {file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")


def main():
    """Точка входа в программу."""
    app = AntennaApp()
    app.mainloop()


if __name__ == '__main__':
    main()