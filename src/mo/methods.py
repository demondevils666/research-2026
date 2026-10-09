"""Методы кластеризации: k-means, KEFRiN (признаки + сеть), Leiden (только сеть), спектральная, CANUS.

KEFRiN (Shalileh & Mirkin 2022, Entropy 24:626) минимизирует
    F = ρ·Σ_i ||y_i − c_k(i)||² + ξ·Σ_i ||p_i − λ_k(i)||²,
где y_i — признаки МО, p_i — его строка сетевой матрицы. Это ровно k-means на склейке Z = [√ρ·Y | √ξ·P], поэтому
считаем его через k-means (k-means++, несколько стартов) — критерий и правило назначения совпадают со статьей.
Предобработка как рекомендует статья: признаки — z-оценки, сеть — «модулярная» стандартизация p_ij − p_i+·p_+j/p_++.
Сетевая матрица P — сходство узлов. Сырые строки смежности при ~5 соседях из 2 016 почти не различают кластеры
(проверено: до α = 0,7 типология не меняется, при α ≥ 0,8 — вырожденные мелкие кластеры), поэтому берем диффузионное
сходство: P = sym((T + T² + … + T^s)/s), T = D⁻¹W — «где окажется путник за 1…s шагов по соседям».
Вес сети задаем одной ручкой α ∈ [0, 1]: ρ = (1−α)/||Y||², ξ = α/||P||², то есть α — доля сети в суммарном разбросе.
α = 0 — обычный k-means по признакам.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering

from .config import ROOT, load


def modularity_standardize(W: np.ndarray) -> np.ndarray:
    tot = W.sum()
    return W - np.outer(W.sum(1), W.sum(0)) / tot if tot > 0 else W.copy()


def diffusion(W: np.ndarray, steps: int) -> np.ndarray:
    """Симметризованное среднее T + T² + … + T^steps, T — матрица случайного блуждания по сети."""
    if steps <= 0:
        return W
    deg = W.sum(1, keepdims=True)
    T = np.divide(W, deg, out=np.zeros_like(W), where=deg > 0)
    acc, Tk = np.zeros_like(W), np.eye(len(W))
    for _ in range(steps):
        Tk = Tk @ T
        acc += Tk
    acc /= steps
    return (acc + acc.T) / 2


def network_matrix(W: np.ndarray, steps: int | None = None) -> np.ndarray:
    """Сетевая матрица KEFRiN: модулярно стандартизованное диффузионное сходство."""
    if steps is None:
        steps = load()["network"]["diffusion_steps"]
    return modularity_standardize(diffusion(W, steps))


def kefrin_space(Y: np.ndarray, W: np.ndarray, alpha: float, steps: int | None = None) -> np.ndarray:
    """Пространство KEFRiN: [√ρ·Y | √ξ·P] при доле сети α; P — диффузионное сходство за steps шагов."""
    return kefrin_Z(Y, network_matrix(W, steps), alpha)


def kefrin_Z(Y: np.ndarray, P: np.ndarray, alpha: float) -> np.ndarray:
    ty, tp = (Y ** 2).sum(), (P ** 2).sum()
    rho, xi = (1 - alpha) / ty, (alpha / tp if tp > 0 else 0.0)
    return np.hstack([np.sqrt(rho) * Y, np.sqrt(xi) * P])


def kefrin(Y: np.ndarray, W: np.ndarray, K: int, alpha: float, seed: int, n_init: int = 10,
           max_iter: int = 300, init=None, steps: int | None = None, Z: np.ndarray | None = None) -> tuple[np.ndarray, KMeans]:
    """KEFRiN (евклидова версия). init — центры в пространстве KEFRiN (для теплого старта); Z — готовое пространство."""
    Z = kefrin_space(Y, W, alpha, steps) if Z is None else Z
    km = KMeans(n_clusters=K, n_init=1 if init is not None else n_init, init=init if init is not None else "k-means++",
                max_iter=max_iter, random_state=seed).fit(Z)
    return km.labels_, km


def kmeans(Y: np.ndarray, K: int, seed: int, n_init: int = 10) -> np.ndarray:
    return KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit(Y).labels_


def leiden_k(edges, ids: list[int], K: int, seed: int, tol_iter: int = 40) -> tuple[np.ndarray, float]:
    """Leiden (модулярность с разрешением γ); γ подбирается бисекцией, чтобы получить K сообществ.

    Если ровно K не достигается, берется ближайшее. Мелкие компоненты связности дают отдельные сообщества.
    """
    import igraph as ig
    import leidenalg as la
    pos = {t: i for i, t in enumerate(ids)}
    g = ig.Graph(n=len(ids), edges=list(zip(edges["a"].map(pos), edges["b"].map(pos))))
    w = edges["w"].tolist()

    def run(gamma):
        p = la.find_partition(g, la.RBConfigurationVertexPartition, weights=w, resolution_parameter=gamma, seed=seed)
        return np.array(p.membership)

    lo, hi = 1e-4, 1.0
    while len(set(run(hi))) < K and hi < 100:
        hi *= 2
    best = None
    for _ in range(tol_iter):
        mid = np.sqrt(lo * hi)
        lab = run(mid)
        n = len(set(lab))
        if best is None or abs(n - K) < abs(len(set(best[0])) - K):
            best = (lab, mid)
        if n == K:
            break
        lo, hi = (mid, hi) if n < K else (lo, mid)
    return best


def spectral(W: np.ndarray, K: int, seed: int) -> np.ndarray:
    return SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed,
                              assign_labels="cluster_qr").fit_predict(W)


def canus(X: np.ndarray, W: np.ndarray, K: int, seed: int, epochs: int = 100, **params) -> np.ndarray:
    """CANUS (Shalileh 2025) — код автора из third_party/canus (scripts/fetch_external.py), без изменений.

    params — параметры CANUSInit автора (learning_rate, attribute_distance, network_distance, rho, zeta, update_rule…).
    """
    p = ROOT / "third_party" / "canus"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import torch
    from canus import CANUSClusterer
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    m = CANUSClusterer(n_clusters=K, epochs=epochs, seed=seed, device="cpu", **params)
    m.fit(X.astype(np.float32), W.astype(np.float32))
    return np.asarray(m.y_pred)


def moved_by_network(lab0: np.ndarray, lab1: np.ndarray, ids: list, edges: pd.DataFrame) -> dict:
    """Что сделала сеть: разбиение без сети (lab0, α = 0) против разбиения с сетью (lab1), метки выровнены.

    n_moved — сколько МО сменили тип. Для них own_before / own_after — средняя доля веса их связей, ведущих в МО
    своего типа, без сети и с сетью: если доля растет, сеть «подклеивает» пограничные МО к соседям.
    """
    pos = {t: i for i, t in enumerate(ids)}
    e = edges[edges["a"].isin(pos) & edges["b"].isin(pos)]
    a, b = e["a"].map(pos).to_numpy(), e["b"].map(pos).to_numpy()
    w, n = e["w"].to_numpy(float), len(ids)
    tot = np.bincount(a, w, n) + np.bincount(b, w, n)

    def own(lab: np.ndarray) -> np.ndarray:
        ws = w * (lab[a] == lab[b])
        return (np.bincount(a, ws, n) + np.bincount(b, ws, n)) / np.where(tot > 0, tot, 1)

    moved = np.flatnonzero(lab0 != lab1)
    return {"n": n, "n_moved": int(len(moved)),
            "own_before": float(own(lab0)[moved].mean()) if len(moved) else None,
            "own_after": float(own(lab1)[moved].mean()) if len(moved) else None}


def align_labels(ref: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Перенумеровать lab так, чтобы метки максимально совпадали с ref (венгерский алгоритм)."""
    from scipy.optimize import linear_sum_assignment
    K = max(ref.max(), lab.max()) + 1
    C = np.zeros((K, K))
    np.add.at(C, (lab, ref), 1)
    r, c = linear_sum_assignment(-C)
    m = dict(zip(r, c))
    return np.array([m.get(x, x) for x in lab])
