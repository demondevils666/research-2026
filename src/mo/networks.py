"""Правила ребра и сети МО.

Главная сеть — физико-функциональная (независима от признаков):
    w_ij = 1[у i и j общая граница] · exp(−d_ij / τ),
d_ij — дорожное расстояние между центрами МО (км), τ — медиана d по соседним парам. У пар «город + окружающий район»
центр общий, d = 0 и вес 1. Если дороги нет, d = расстояние по прямой × медианная извилистость. МО без соседей в панели
связываются с несколькими ближайшими по дороге по той же формуле.

Сети сравнения (kNN, симметризация «или»): косинус профиля, корреляция помесячных изменений уровня,
максимальная лаговая корреляция, DTW относительного уровня, ближайшие по дороге.
Сеть хранится как таблица ребер (a < b, w) в порядке узлов `ids`.
"""

import numpy as np
import pandas as pd

from . import data
from .config import load


def _road_km(ids: list[int]) -> dict:
    con = data.hack("connection")
    con = con[(con["type"] == "highway") & con["territory_id_x"].isin(ids) & con["territory_id_y"].isin(ids)]
    a, b = np.minimum(con["territory_id_x"], con["territory_id_y"]), np.maximum(con["territory_id_x"], con["territory_id_y"])
    return dict(zip(zip(a, b), con["distance"]))


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def _centers(mo: pd.DataFrame) -> pd.DataFrame:
    """Центры МО; для городов федерального значения (нет координат центра) — центроид полигона."""
    c = mo[["lat", "lon"]].copy()
    miss = c["lat"].isna() | c["lon"].isna()
    if miss.any():
        g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last").set_index("territory_id")
        cen = g.loc[c.index[miss], "geometry"].to_crs(3857).centroid.to_crs(4326)
        c.loc[miss, "lat"], c.loc[miss, "lon"] = cen.y.to_numpy(), cen.x.to_numpy()
    return c


def edge_km(edges: pd.DataFrame, mo: pd.DataFrame, road: dict | None = None) -> np.ndarray:
    """Расстояние по ребру: дорожное, иначе по прямой × извилистость."""
    road = road if road is not None else _road_km(sorted(mo.index))
    det = load()["network"]["detour"]
    c = _centers(mo)
    d = np.array([road.get((a, b), np.nan) for a, b in zip(edges["a"], edges["b"])], dtype=float)
    miss = np.isnan(d)
    if miss.any():
        ea, eb = edges["a"].to_numpy()[miss], edges["b"].to_numpy()[miss]
        d[miss] = det * haversine_km(c.loc[ea, "lat"].to_numpy(), c.loc[ea, "lon"].to_numpy(),
                                     c.loc[eb, "lat"].to_numpy(), c.loc[eb, "lon"].to_numpy())
    return d


def physical(mo: pd.DataFrame, cont: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Главная сеть: соседство по границе с гравитационным весом + связи изолятов по дороге."""
    cfg = load()["network"]
    ids = sorted(mo.index)
    road = _road_km(ids)
    e = cont[["a", "b"]].copy()
    e["km"] = edge_km(e, mo, road)
    tau = float(np.median(e["km"])) if cfg["tau"] == "auto" else float(cfg["tau"])
    e["kind"] = "border"
    deg = pd.concat([e["a"], e["b"]]).value_counts().reindex(ids).fillna(0)
    iso = [t for t in ids if deg[t] == 0]
    if iso:
        r = pd.DataFrame([(a, b, d) for (a, b), d in road.items()], columns=["a", "b", "km"])
        both = pd.concat([r.rename(columns={"a": "src", "b": "dst"}), r.rename(columns={"b": "src", "a": "dst"})])
        near = both[both["src"].isin(iso)].sort_values(["src", "km"]).groupby("src").head(cfg["isolated_road_k"])
        extra = pd.DataFrame({"a": np.minimum(near["src"], near["dst"]), "b": np.maximum(near["src"], near["dst"]),
                              "km": near["km"], "kind": "road_isolated"})
        e = pd.concat([e, extra]).drop_duplicates(["a", "b"])
    e["w"] = np.exp(-e["km"] / tau)
    return e.reset_index(drop=True), {"tau_km": tau, "isolated_linked_by_road": len(iso)}


def knn_from_similarity(S: np.ndarray, ids: list[int], k: int, weight: str = "sim") -> pd.DataFrame:
    """kNN по матрице сходства (больше — ближе), симметризация «или». weight: 'sim' (сходство ≥ 0) или 'one'."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(ids)), k)
    cols = nn.ravel()
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    e = pd.DataFrame({"i": a, "j": b}).drop_duplicates()
    w = S[e["i"], e["j"]] if weight == "sim" else np.ones(len(e))
    idx = np.asarray(ids)
    return pd.DataFrame({"a": idx[e["i"]], "b": idx[e["j"]], "w": np.clip(w, 1e-6, None)}).reset_index(drop=True)


def gaussian_knn(X: np.ndarray, ids: list[int], k: int) -> pd.DataFrame:
    """kNN по признакам с самонастраивающимся гауссовым ядром exp(−d²/(σ_i σ_j)), σ_i — расстояние до k-го соседа."""
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    sig = np.sort(D, axis=1)[:, k]
    S = np.exp(-(D ** 2) / np.outer(sig, sig).clip(1e-12))
    return knn_from_similarity(S, ids, k)


def cosine_sim(X: np.ndarray) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    return Xn @ Xn.T


def corr_sim(R: np.ndarray) -> np.ndarray:
    """Корреляция Пирсона строк R (ряды МО)."""
    Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True).clip(1e-12)
    return Z @ Z.T / R.shape[1]


def lag_corr_sim(R: np.ndarray, lag_max: int) -> np.ndarray:
    """Максимум по лагам |l| <= lag_max корреляции рядов (по пересечению)."""
    best = np.full((R.shape[0], R.shape[0]), -np.inf)
    T = R.shape[1]
    for l in range(-lag_max, lag_max + 1):
        A = R[:, max(0, l):T + min(0, l)]
        B = R[:, max(0, -l):T + min(0, -l)]
        Za = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True).clip(1e-12)
        Zb = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True).clip(1e-12)
        best = np.maximum(best, Za @ Zb.T / A.shape[1])
    return np.maximum(best, best.T)


def dtw_dist(R: np.ndarray, window: int, chunk: int = 200_000) -> np.ndarray:
    """Матрица DTW-расстояний между рядами (одинаковой длины) с окном Сакоэ–Тибы; векторно по парам."""
    n, T = R.shape
    iu, ju = np.triu_indices(n, 1)
    out = np.zeros((n, n))
    for s in range(0, len(iu), chunk):
        a, b = R[iu[s:s + chunk]], R[ju[s:s + chunk]]
        D = {}
        for i in range(T):
            for j in range(max(0, i - window), min(T, i + window + 1)):
                cost = (a[:, i] - b[:, j]) ** 2
                if i == 0 and j == 0:
                    prev = 0.0
                else:
                    cand = [D[k] for k in ((i - 1, j), (i, j - 1), (i - 1, j - 1)) if k in D]
                    prev = np.minimum.reduce(cand) if len(cand) > 1 else cand[0]
                D[(i, j)] = cost + prev
        v = np.sqrt(D[(T - 1, T - 1)])
        out[iu[s:s + chunk], ju[s:s + chunk]] = v
    return out + out.T


def to_dense(edges: pd.DataFrame, ids: list[int]) -> np.ndarray:
    pos = {t: i for i, t in enumerate(ids)}
    W = np.zeros((len(ids), len(ids)))
    ia, ib = edges["a"].map(pos).to_numpy(), edges["b"].map(pos).to_numpy()
    W[ia, ib] = edges["w"].to_numpy()
    W[ib, ia] = edges["w"].to_numpy()
    return W


def stats(edges: pd.DataFrame, mo: pd.DataFrame, ring_pairs: pd.DataFrame, road: dict | None = None) -> dict:
    """Описание сети: размер, связность, «внутри региона», длина ребер, транзитивность, покрытие пар «город + район»."""
    import networkx as nx
    ids = sorted(mo.index)
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(zip(edges["a"], edges["b"]))
    deg = np.array([d for _, d in g.degree()])
    reg = mo["region_code"]
    km = edge_km(edges, mo, road)
    E = set(zip(edges["a"], edges["b"]))
    rp = {(min(r, c), max(r, c)) for r, c in zip(ring_pairs["ring"], ring_pairs["city"])}
    return {"edges": int(len(edges)), "mean_degree": round(float(deg.mean()), 2), "isolated": int((deg == 0).sum()),
            "components": nx.number_connected_components(g),
            "share_within_region": round(float((reg.loc[edges["a"]].to_numpy() == reg.loc[edges["b"]].to_numpy()).mean()), 3),
            "edge_km_median": round(float(np.median(km)), 1),
            "transitivity": round(float(nx.transitivity(g)), 3),
            "ring_pairs_covered": round(len(E & rp) / len(rp), 3)}


def jaccard(e1: pd.DataFrame, e2: pd.DataFrame) -> float:
    a, b = set(zip(e1["a"], e1["b"])), set(zip(e2["a"], e2["b"]))
    return len(a & b) / len(a | b) if a | b else 0.0


def monthly_layer(edges: pd.DataFrame, ids: list[int], Ym: dict, h: float | str = "auto") -> tuple[dict, float]:
    """Помесячный слой сети: топология и вес физической сети, умноженный на сходство соседей в этом месяце.

    W_t(i, j) = w_phys(i, j) · exp(−‖x_i(t) − x_j(t)‖² / 2h²), x(t) — признаки месяца, стандартизованные внутри месяца.
    h — медиана расстояний по ребрам за все месяцы ("auto") или число: одна шкала для всех месяцев, поэтому изменения
    среднего веса — это сближение или расхождение соседей. При h → ∞ слой совпадает с физической сетью.
    Возвращает {месяц: ребра a, b, w, sim} и h.
    """
    pos = {t: i for i, t in enumerate(ids)}
    ia, ib = edges["a"].map(pos).to_numpy(), edges["b"].map(pos).to_numpy()
    dist = {d: np.linalg.norm(Y[ia] - Y[ib], axis=1) for d, Y in Ym.items()}
    if h == "auto":
        h = float(np.median(np.concatenate(list(dist.values()))))
    out = {}
    for d, dd in dist.items():
        sim = np.exp(-dd ** 2 / (2 * h ** 2))
        out[d] = pd.DataFrame({"a": edges["a"].to_numpy(), "b": edges["b"].to_numpy(),
                               "w": edges["w"].to_numpy() * sim, "sim": sim})
    return out, float(h)
