"""Проба P3 (этап 1): «пояс или архипелаг» — как типы МО по тратам ложатся на географию.

Быстрая типология только для пробы (не финальная модель): k-means по годовому профилю 2024
(CLR шести долей + относительный уровень, стандартизовано), K из configs/base.yaml (probes.type_k).
На графе соседства по границе считаем для каждого типа:
  same_edge_share — доля ребер типа, ведущих к МО того же типа;
  ожидание при случайной перестановке меток (размеры типов сохраняются) и z-оценка;
  has_same_neighbor — доля МО типа, у которых есть сосед того же типа.
«Пояс» — тип сильно сцеплен в пространстве (z велика), «архипелаг» — похожие МО разбросаны (z ≈ 0).
Выход: outputs/probes/p3_belt_archipelago.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mo import features  # noqa: E402
from mo.spatial import assortativity, type_stats  # noqa: E402
from mo.config import load, path  # noqa: E402
from mo.agglo import units  # noqa: E402


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "probes"
    mo = pd.read_parquet(proc / "mo.parquet")
    prof = pd.read_parquet(proc / "profile.parquet")
    X = features.standardize(prof)
    cont = pd.read_parquet(proc / "edges_contiguity.parquet")
    pos = {t: i for i, t in enumerate(X.index)}
    A_u, A_v = cont["a"].map(pos).to_numpy(), cont["b"].map(pos).to_numpy()
    w = pd.read_parquet(proc / "panel_wide.parquet")
    wy = w[w.index.get_level_values("date").str.startswith(cfg["features"]["profile_year"])].groupby(level="territory_id").mean()
    sh = features.shares(wy).loc[X.index]
    u = units(mo, pd.read_parquet(proc / "ring_pairs.parquet"), cont)
    res = {"definition": __doc__.split("Выход")[0].strip(), "n_mo": len(X), "contiguity_edges": len(cont)}
    for K in cfg["probes"]["type_k"]:
        km = KMeans(n_clusters=K, n_init=20, random_state=cfg["seed"]).fit(X.to_numpy())
        lab = km.labels_
        ts = type_stats(lab, A_u, A_v, cfg["probes"]["permutations"] // 4, cfg["seed"])
        # Профиль типа для узнавания: медианные доли, траты на жителя, доля горожан, примеры
        prof_rows = []
        for k in range(K):
            ids = X.index[lab == k]
            top = mo.loc[ids].sort_values("population", ascending=False)["name"].head(4).tolist()
            prof_rows.append({"type": k,
                              **{f"share_{c}": round(float(sh.loc[ids, c].median()) * 100, 1) for c in sh.columns},
                              "spend_rub_median": int(wy.loc[ids, cfg["panel"]["total_category"]].median()),
                              "urban_share_median": round(float(mo.loc[ids, "urban_share"].median()), 2),
                              "wage_median": int(mo.loc[ids, "wage"].median()),
                              "n_regions": int(mo.loc[ids, "region"].nunique()),
                              "examples": top})
        # Без внутригородских территорий Москвы и СПб: они граничат друг с другом и завышают сцепленность
        keep = ~mo.loc[X.index, "type"].str.startswith("внутригородская").to_numpy()
        e_ok = keep[A_u] & keep[A_v]
        remap = -np.ones(len(lab), dtype=int)
        remap[keep] = np.arange(keep.sum())
        ts2 = type_stats(lab[keep], remap[A_u[e_ok]], remap[A_v[e_ok]], cfg["probes"]["permutations"] // 4, cfg["seed"])
        ts = ts.merge(ts2[["type", "n", "same_edge_share", "z", "lift"]].add_suffix("_no_fed_cities")
                      .rename(columns={"type_no_fed_cities": "type"}), on="type")
        lab_s = pd.Series(lab, index=X.index)
        same_type = {g: round(float((lab_s.loc[uu["unit"]].to_numpy() == lab_s.loc[uu["partner"]].to_numpy()).mean()), 3)
                     for g, uu in u.groupby("group")}
        res[f"K{K}"] = {"assortativity": round(assortativity(lab, A_u, A_v), 3),
                        "share_same_type_as_city": same_type,
                        "assortativity_no_fed_cities": round(assortativity(lab[keep], remap[A_u[e_ok]], remap[A_v[e_ok]]), 3),
                        "types": ts.merge(pd.DataFrame(prof_rows), on="type").to_dict("records")}
    (out / "p3_belt_archipelago.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for K in cfg["probes"]["type_k"]:
        r = res[f"K{K}"]
        print(f"K={K} assortativity={r['assortativity']} (без районов Москвы и СПб: {r['assortativity_no_fed_cities']}); "
              f"тот же тип, что у города: {r['share_same_type_as_city']}")
        print(pd.DataFrame(r["types"])[["type", "n", "same_edge_share", "expected_random", "z", "lift", "lift_no_fed_cities", "n_no_fed_cities", "has_same_neighbor",
                                         "spend_rub_median", "share_Общественное питание", "share_Маркетплейсы",
                                         "urban_share_median", "n_regions", "examples"]].to_string(index=False))


if __name__ == "__main__":
    main()
