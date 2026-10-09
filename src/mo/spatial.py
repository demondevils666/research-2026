"""Пространственная структура типов: «пояс или архипелаг».

Для каждого типа на графе соседства по границе: доля ребер типа, ведущих к МО того же типа, против ожидания при случайной
перестановке меток (размеры типов сохраняются), z-оценка и lift (во сколько раз чаще случайного). Плюс номинальная
ассортативность Ньюмана по всем типам.
"""

import numpy as np
import pandas as pd


def type_stats(lab: np.ndarray, A_u: np.ndarray, A_v: np.ndarray, n_perm: int, seed: int) -> pd.DataFrame:
    K = lab.max() + 1

    def same_share(l):
        lu, lv = l[A_u], l[A_v]
        same = lu == lv
        out = np.zeros(K)
        for k in range(K):
            ends = (lu == k).sum() + (lv == k).sum()
            out[k] = 2 * (same & (lu == k)).sum() / ends if ends else np.nan
        return out

    obs = same_share(lab)
    rng = np.random.default_rng(seed)
    perm = np.array([same_share(rng.permutation(lab)) for _ in range(n_perm)])
    n = len(lab)
    nb_same = np.zeros(n, dtype=bool)
    same = lab[A_u] == lab[A_v]
    nb_same[A_u[same]] = True
    nb_same[A_v[same]] = True
    has_edge = np.zeros(n, dtype=bool)
    has_edge[A_u] = True
    has_edge[A_v] = True
    rows = []
    for k in range(K):
        mu, sd = perm[:, k].mean(), perm[:, k].std()
        m = (lab == k) & has_edge
        rows.append({"type": k, "n": int((lab == k).sum()), "same_edge_share": round(float(obs[k]), 3),
                     "expected_random": round(float(mu), 3), "z": round(float((obs[k] - mu) / sd), 1),
                     "lift": round(float(obs[k] / mu), 2), "has_same_neighbor": round(float(nb_same[m].mean()), 3)})
    return pd.DataFrame(rows)


def assortativity(lab: np.ndarray, A_u: np.ndarray, A_v: np.ndarray) -> float:
    """Номинальная ассортативность Ньюмана по типам на неориентированном графе."""
    K = lab.max() + 1
    e = np.zeros((K, K))
    np.add.at(e, (lab[A_u], lab[A_v]), 1)
    np.add.at(e, (lab[A_v], lab[A_u]), 1)
    e /= e.sum()
    a = e.sum(axis=1)
    return float((np.trace(e) - (a * a).sum()) / (1 - (a * a).sum()))
