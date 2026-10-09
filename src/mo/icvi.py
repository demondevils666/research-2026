"""Внутренние индексы качества кластеризации (ICVI) в пространстве признаков и в сети.

Признаки: SW (силуэт), CH (Калински–Харабаш), S_Dbw (Halkidi & Vazirgiannis 2001; меньше — лучше).
Сеть (по симметричной взвешенной матрице W без петель; S_ab — сумма весов между кластерами a и b, при a = b
каждое внутреннее ребро учтено дважды):
  AVI   = (1/K) Σ_a S_aa / Σ_b S_ab                                  — средняя изолированность, больше — лучше;
  AVU   = (1/K) Σ_a Σ_{b≠a} S_ab / (out_a + out_b − S_ab)             — средняя «слитость», меньше — лучше;
  ANUI  = 1 / (AVU + 1/AVI);
  Q     = Σ_a [S_aa/(2m) − (d_a/2m)²]                                 — модулярность Ньюмана (основная версия MQ);
  Qdens = Σ_a (S_aa/2 − d_a²/(4m)) / n_a                              — density modularity (как в Pattern);
  TurboMQ = Σ_a CF_a, CF_a = 2μ_a / (2μ_a + Σ_{b≠a}(ε_ab + ε_ba)) (Mitchell & Mancoridis 2006). Для неориентированного
            графа CF_a совпадает с изолированностью, поэтому TurboMQ = K·AVI — считаем, но в выводах не дублируем;
  BasicMQ = (1/K) Σ_a μ_a/n_a² − (2/(K(K−1))) Σ_{a<b} ε_ab/(2 n_a n_b) (Mancoridis et al. 1998).
AVI, AVU, ANUI, Q, Qdens — по формулам библиотеки Pattern лаборатории жюри (сверено в tests/test_icvi.py).
"""

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score

FEATURE_KEYS = ["SW", "CH", "S_Dbw"]
GRAPH_KEYS = ["AVI", "AVU", "ANUI", "Q", "Qdens", "TurboMQ", "BasicMQ"]
HIGHER_BETTER = {"SW": True, "CH": True, "S_Dbw": False, "AVI": True, "AVU": False, "ANUI": True, "Q": True,
                 "Qdens": True, "TurboMQ": True, "BasicMQ": True}


def _codes(labels) -> tuple[np.ndarray, int]:
    _, y = np.unique(np.asarray(labels), return_inverse=True)
    return y, int(y.max()) + 1


def s_dbw(X: np.ndarray, labels, variant: str = "paper") -> float:
    """S_Dbw = Scat + Dens_bw (Halkidi & Vazirgiannis, ICDM 2001).

    variant="paper": σ — вектор дисперсий (как в статье), плотность у центров считается по точкам обоих кластеров.
    variant="package": как в пакете s-dbw (method='Halkidi'): σ — вектор стандартных отклонений, плотность у центра
    кластера — только по его собственным точкам. Значения вариантов различаются; основной — "paper".
    """
    y, K = _codes(labels)
    centers = np.array([X[y == k].mean(0) for k in range(K)])
    disp = (lambda Z: Z.var(0)) if variant == "paper" else (lambda Z: Z.std(0))
    sig_all = np.linalg.norm(disp(X))
    sig_k = np.array([np.linalg.norm(disp(X[y == k])) for k in range(K)])
    scat = float(np.mean(sig_k / sig_all))
    stdev = np.sqrt(sig_k.sum()) / K

    def density(u, pts):
        return int((np.linalg.norm(pts - u, axis=1) <= stdev).sum())

    tot = 0.0
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            pts = X[(y == a) | (y == b)]
            u = (centers[a] + centers[b]) / 2
            if variant == "paper":
                den = max(density(centers[a], pts), density(centers[b], pts))
            else:
                den = max(density(centers[a], X[y == a]), density(centers[b], X[y == b]))
            tot += density(u, pts) / den if den > 0 else 0.0
    return scat + tot / (K * (K - 1))


def feature_indices(X: np.ndarray, labels) -> dict:
    y, K = _codes(labels)
    if K < 2:
        return {k: np.nan for k in FEATURE_KEYS}
    return {"SW": float(silhouette_score(X, y)), "CH": float(calinski_harabasz_score(X, y)), "S_Dbw": float(s_dbw(X, y))}


def graph_indices(W: np.ndarray, labels) -> dict:
    y, K = _codes(labels)
    H = np.eye(K)[y]
    S = H.T @ W @ H
    tot, d = S.sum(1), np.diag(S)
    n = H.sum(0)
    out = tot - d
    with np.errstate(divide="ignore", invalid="ignore"):
        avi = float(np.mean(np.where(tot > 0, d / tot, 0.0)))
        den = out[:, None] + out[None, :] - S
        U = np.where(den != 0, S / den, 0.0)
    np.fill_diagonal(U, 0.0)
    avu = float(U.sum() / K)
    anui = 1.0 / (avu + 1.0 / avi) if avi > 0 else 0.0
    two_m = W.sum()
    q = float(np.sum(d / two_m - (tot / two_m) ** 2)) if two_m > 0 else 0.0
    qd = float(np.sum((d / 2 - tot ** 2 / (2 * two_m)) / n)) if two_m > 0 else 0.0
    mu = d / 2
    turbo = float(np.sum(np.where(2 * mu + out > 0, 2 * mu / (2 * mu + out), 0.0)))
    intra = float(np.mean(mu / n ** 2))
    if K > 1:
        iu = np.triu_indices(K, 1)
        inter = float(np.sum(S[iu] / (2 * np.outer(n, n)[iu])) * 2 / (K * (K - 1)))
    else:
        inter = 0.0
    return {"AVI": avi, "AVU": avu, "ANUI": anui, "Q": q, "Qdens": qd, "TurboMQ": turbo, "BasicMQ": intra - inter}


def all_indices(X: np.ndarray, W: np.ndarray | dict, labels) -> dict:
    """Все индексы. W — одна матрица или словарь {имя сети: матрица} (сетевые индексы с суффиксом сети)."""
    res = feature_indices(X, labels)
    nets = W if isinstance(W, dict) else {"": W}
    for name, M in nets.items():
        g = graph_indices(M, labels)
        res.update({(f"{k}@{name}" if name else k): v for k, v in g.items()})
    return res


def random_baseline(X: np.ndarray, W, labels, n: int, seed: int) -> dict:
    """Среднее и стандартное отклонение индексов при случайной перестановке меток (размеры кластеров сохраняются)."""
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    vals = [all_indices(X, W, rng.permutation(lab)) for _ in range(n)]
    keys = vals[0].keys()
    return {k: {"mean": float(np.mean([v[k] for v in vals])), "std": float(np.std([v[k] for v in vals]))} for k in keys}
