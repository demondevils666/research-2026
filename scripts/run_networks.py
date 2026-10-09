"""Этап 2, шаг 1: правила ребра и сравнение сетей.

Строит главную физико-функциональную сеть и сети сравнения (kNN по сходству), описывает каждую
и считает пересечение ребер между правилами.
Выход: data/processed/net_<rule>.parquet (ребра a, b, w), outputs/stage2/networks.json
Запуск: python scripts/run_networks.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, networks  # noqa: E402
from mo.config import load, path  # noqa: E402

RULES = {
    "physical": "граница × exp(−км/τ), пары «город + район» вес 1 (главная, независима от признаков)",
    "cosine": "kNN по косинусу годового профиля 2024 (структура + уровень)",
    "corr": "kNN по корреляции помесячных изменений относительного уровня трат",
    "lagcorr": "kNN по максимуму лаговой корреляции изменений уровня (|лаг| ≤ 2)",
    "dtw": "kNN по DTW нормированного ряда относительного уровня (окно 2)",
    "road": "kNN ближайших по дороге, вес exp(−км/τ)",
}


def main() -> None:
    cfg = load()
    k = cfg["network"]["knn_k"]
    proc, out = path("processed"), path("outputs") / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    cont = pd.read_parquet(proc / "edges_contiguity.parquet")
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    prof = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids]
    fm = pd.read_parquet(proc / "features_monthly.parquet")
    lvl = fm["rel_level"].unstack("date").loc[ids].to_numpy()          # МО × 24
    dl = np.diff(lvl, axis=1)                                            # помесячные изменения
    z = (lvl - lvl.mean(1, keepdims=True)) / lvl.std(1, keepdims=True).clip(1e-12)

    res = {"rules": RULES, "knn_k": k, "networks": {}, "timing_s": {}}
    edges = {}
    t = time.time()
    edges["physical"], info = networks.physical(mo, cont)
    res["physical_info"] = info
    res["timing_s"]["physical"] = round(time.time() - t, 1)
    tau = info["tau_km"]
    sims = {
        "cosine": lambda: networks.cosine_sim(prof.to_numpy()),
        "corr": lambda: networks.corr_sim(dl),
        "lagcorr": lambda: networks.lag_corr_sim(dl, cfg["network"]["lag_max"]),
        "dtw": lambda: -networks.dtw_dist(z, cfg["network"]["dtw_window"]),
    }
    for name, f in sims.items():
        t = time.time()
        S = f()
        e = networks.knn_from_similarity(S, ids, k, weight="sim" if name != "dtw" else "one")
        if name == "dtw":  # вес из расстояния: exp(−d/медиана d по ребрам)
            pos = {t_: i for i, t_ in enumerate(ids)}
            d = -S[e["a"].map(pos), e["b"].map(pos)]
            e["w"] = np.exp(-d / np.median(d))
        edges[name] = e
        res["timing_s"][name] = round(time.time() - t, 1)
    from mo import links
    rk = links.road_knn(set(ids), k)
    rk["w"] = np.exp(-rk["km"] / tau)
    edges["road"] = rk[["a", "b", "w"]]

    road = networks._road_km(ids)
    for name, e in edges.items():
        res["networks"][name] = networks.stats(e, mo, rp, road)
        e[["a", "b", "w"]].to_parquet(proc / f"net_{name}.parquet")
    names = list(edges)
    res["jaccard"] = {a: {b: round(networks.jaccard(edges[a], edges[b]), 3) for b in names} for a in names}
    (out / "networks.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(pd.DataFrame(res["networks"]).T.to_string())
    print(pd.DataFrame(res["jaccard"]).to_string())
    print(res["physical_info"], res["timing_s"])


if __name__ == "__main__":
    main()
