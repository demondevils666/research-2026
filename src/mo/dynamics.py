"""Отслеживание типов во времени: эволюционный KEFRiN.

На каждом месяце t признаки сглаживаются с историей Ỹ_t = β·Ỹ_{t−1} + (1−β)·Y_t (временная гладкость, как в
evolutionary clustering, Chakrabarti et al. 2006), сеть постоянна. Кластеризация KEFRiN в пространстве
[√ρ·Ỹ_t | √ξ·P] с теплым стартом от центров прошлого месяца: номера типов сохраняют смысл. Первый месяц стартует
от центров статической типологии (годовой профиль 2024), поэтому типы помесячной модели — те же типы.
Номера типов каждого месяца сопоставляются со статической типологией венгерским алгоритмом (максимум общих МО).
Отслеживание самих типов — внешние переходы MONIC (функция monic).
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import methods


def per_month_standardized(fm: pd.DataFrame) -> dict[str, np.ndarray]:
    """Признаки по месяцам, стандартизованные по МО внутри месяца (относительно страны в этом месяце)."""
    out = {}
    for date, g in fm.groupby(level="date"):
        x = g.droplevel("date").sort_index()
        out[date] = ((x - x.mean()) / x.std(ddof=0)).to_numpy()
    return out


def centers_from_labels(Z: np.ndarray, lab: np.ndarray, K: int) -> np.ndarray:
    return np.array([Z[lab == k].mean(0) for k in range(K)])


def evolve(Ym: dict, P: np.ndarray | dict, static: np.ndarray, K: int, alpha: float, beta: float, seed: int) -> dict:
    """Эволюционный KEFRiN по месяцам. Возвращает метки МО × месяц и метрики.

    P — сетевая матрица KEFRiN (постоянная сеть) или словарь {месяц: P_t} (сеть по месяцам, этап 3b).
    """
    dates = sorted(Ym)
    labs, sw = {}, []
    Ys = None
    init = None
    for d in dates:
        Ys = Ym[d] if Ys is None else beta * Ys + (1 - beta) * Ym[d]
        Z = methods.kefrin_Z(Ys, P[d] if isinstance(P, dict) else P, alpha)
        if init is None:
            init = centers_from_labels(Z, static, K)
        km = KMeans(n_clusters=K, init=init, n_init=1, max_iter=300, random_state=seed).fit(Z)
        lab = methods.align_labels(static, km.labels_)
        init = centers_from_labels(Z, lab, K)
        labs[d] = lab
        sw.append(silhouette_score(Ym[d], lab))
    return {"labels": labs, "sw_month": sw}


def independent(Ym: dict, P: np.ndarray, static: np.ndarray, K: int, alpha: float, seed: int, n_init: int) -> dict:
    """Базовая линия: независимая кластеризация каждого месяца + венгерское выравнивание к статической типологии."""
    labs, sw = {}, []
    for d in sorted(Ym):
        lab = methods.kefrin(Ym[d], None, K, alpha, seed, n_init, Z=methods.kefrin_Z(Ym[d], P, alpha))[0]
        lab = methods.align_labels(static, lab)
        labs[d] = lab
        sw.append(silhouette_score(Ym[d], lab))
    return {"labels": labs, "sw_month": sw}


def summary(labs: dict, sw: list, static: np.ndarray, K: int, min_months: int) -> dict:
    dates = sorted(labs)
    L = np.array([labs[d] for d in dates])            # месяц × МО
    changed = (L[1:] != L[:-1]).mean(1)
    T = np.zeros((K, K))
    for a, b in zip(L[:-1], L[1:]):
        np.add.at(T, (a, b), 1)
    T = T / T.sum(1, keepdims=True)
    # «Событие» — смена типа, после которой новый тип держится не меньше min_months месяцев
    n_events, ev_rows = 0, []
    for i in range(L.shape[1]):
        s = L[:, i]
        for t in range(1, len(s) - min_months + 1):
            if s[t] != s[t - 1] and (s[t:t + min_months] == s[t]).all():
                n_events += 1
                ev_rows.append((i, dates[t], int(s[t - 1]), int(s[t])))
    return {
        "sw_month_mean": float(np.mean(sw)),
        "changed_share_mean": float(changed.mean()),
        "changed_share_by_month": dict(zip(dates[1:], np.round(changed, 4).tolist())),
        "never_changed_share": float((L == L[0]).all(0).mean()),
        "ari_consecutive_mean": float(np.mean([adjusted_rand_score(a, b) for a, b in zip(L[:-1], L[1:])])),
        "ari_with_static_mean": float(np.mean([adjusted_rand_score(static, x) for x in L])),
        "transition_matrix": np.round(T, 4).tolist(),
        "n_events": n_events, "events": ev_rows,
    }


def monic(prev: np.ndarray, nxt: np.ndarray, match: float, split: float) -> dict:
    """Внешние переходы типов между двумя срезами по схеме MONIC (Spiliopoulou et al., KDD 2006).

    Перекрытие типа X среза t с типом Y среза t+1 — доля МО типа X, оказавшихся в Y: ov(X, Y) = |X ∩ Y| / |X|.
    X «выживает» в Y, если ov(X, Y) ≥ match и в Y не выживает другой тип; если в один Y выживают два типа и больше —
    это «поглощение» (слияние). X «расщепляется», если ни один Y не берет ≥ match, но типы с ov ≥ split вместе берут
    ≥ match. Иначе X «исчезает». Тип Y «рождается», если в него не переходит ни один тип прошлого среза.
    """
    ks_prev, ks_next = np.unique(prev), np.unique(nxt)
    ov = {x: {y: float(((prev == x) & (nxt == y)).sum() / (prev == x).sum()) for y in ks_next} for x in ks_prev}
    surv, splits, gone = {}, {}, []
    for x in ks_prev:
        best = max(ov[x], key=ov[x].get)
        if ov[x][best] >= match:
            surv[x] = best
            continue
        parts = [y for y in ks_next if ov[x][y] >= split]
        if len(parts) >= 2 and sum(ov[x][y] for y in parts) >= match:
            splits[x] = parts
        else:
            gone.append(x)
    hits = Counter(surv.values())
    reached = set(surv.values()) | {y for ys in splits.values() for y in ys}
    return {"survive": {int(x): int(y) for x, y in surv.items() if hits[y] == 1},
            "absorb": {int(y): sorted(int(x) for x in surv if surv[x] == y) for y, n in hits.items() if n >= 2},
            "split": {int(x): [int(y) for y in ys] for x, ys in splits.items()},
            "disappear": [int(x) for x in gone],
            "emerge": [int(y) for y in ks_next if y not in reached],
            "overlap": {int(x): round(max(ov[x].values()), 4) for x in ks_prev}}
