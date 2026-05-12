import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit  # sigmoid
from scipy.stats import norm

pd.set_option("display.max_columns", 100)

sns.set_theme(style="whitegrid")

RANDOM_STATE = 42

def sample_categorical(rng, categories, probabilities, size):
    """
    Удобная обёртка для генерации категориальных признаков.
    """
    return rng.choice(categories, size=size, p=probabilities)

def generate_correlated_normals(rng, n, rho):
    """
    Генерирует две стандартные нормальные случайные величины с корреляцией rho.

    Parameters
    ----------
    rng : np.random.Generator
        Генератор случайных чисел.
    n : int
        Количество наблюдений.
    rho : float
        Желаемая корреляция между переменными. Должна быть в [-1, 1].

    Returns
    -------
    z1, z2 : np.ndarray
        Два массива длины n с примерно заданной корреляцией.
    """
    if not -1 <= rho <= 1:
        raise ValueError("rho must be between -1 and 1")

    z1 = rng.normal(size=n)
    z2_independent = rng.normal(size=n)

    z2 = rho * z1 + np.sqrt(1 - rho**2) * z2_independent

    return z1, z2
