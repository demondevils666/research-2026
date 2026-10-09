"""Простые статистические проверки: бутстрап-интервал и перестановочный тест разницы двух групп."""

import numpy as np


def bootstrap_ci(x, stat=np.mean, n: int = 2000, seed: int = 42, alpha: float = 0.05) -> list[float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    vals = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


def perm_test_diff(a, b, stat=np.mean, n: int = 2000, seed: int = 42) -> dict:
    """Двусторонний перестановочный тест: stat(a) − stat(b) против перемешивания меток групп."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    obs = stat(a) - stat(b)
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        d = stat(pool[:len(a)]) - stat(pool[len(a):])
        cnt += abs(d) >= abs(obs) - 1e-12
    return {"diff": float(obs), "p_value": float((cnt + 1) / (n + 1)), "n_perm": n}
