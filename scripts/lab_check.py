"""Этап 4: сверка с работами лаборатории СберИндекса (источники — раздел «Литература» отчета).

1. Пригороды, которые назвала сама лаборатория (configs/base.yaml: lab.named_suburbs; отчет о ДФО 4.09.2025): есть ли
   они в нашей базе, их тип и тип их города, группа и индекс спутника (outputs/stage3/satellites.csv). База
   сравнения — доля районов и муниципальных округов в «городских» типах (lab.city_types).
2. Разложение индекса Тейла трат на жителя за lab.year (все категории, среднее за месяцы) на межгрупповую и
   внутригрупповую части. Группы — наши типы, регионы и столько же групп по одному уровню трат (верхняя граница «по
   построению»: уровень трат — признак типологии). Два варианта: МО как единицы (как у лаборатории: «популяция
   состоит из муниципалитетов») и с весом населения (неравенство между жителями). У лаборатории между кластерами 70%,
   между регионами 15%; их 20 кластеров построены по структурным признакам, траты — с поправкой на стоимость жизни,
   у нас траты номинальные.
3. Рост трат на жителя за lab.year к прошлому году: «бублики» против своих городов (парный тест Уилкоксона) и против
   обычных соседей города (перестановочный тест медиан). У лаборатории «быстрорастущие пригороды агломераций» — самый
   быстрый рост 1К24/1К23 среди 20 кластеров.
Выход: outputs/stage3/lab_check.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import census, features, stats  # noqa: E402
from mo.config import load, path  # noqa: E402

DISTRICTS = ("муниципальный район", "муниципальный округ")


def theil_split(y: pd.Series, g: pd.Series, w: pd.Series) -> dict:
    """Индекс Тейла T и доля межгрупповой части: T = Σ s_i ln(y_i/μ), s_i — доля i в сумме трат (с весом w)."""
    d = pd.DataFrame({"y": y, "g": g, "w": w}).dropna()
    d["wy"] = d["w"] * d["y"]
    mu = d["wy"].sum() / d["w"].sum()
    T = float((d["wy"] / d["wy"].sum() * np.log(d["y"] / mu)).sum())
    grp = d.groupby("g")[["wy", "w"]].sum()
    Tb = float((grp["wy"] / d["wy"].sum() * np.log(grp["wy"] / grp["w"] / mu)).sum())
    return {"theil": round(T, 4), "between_share": round(Tb / T, 3), "n_groups": int(len(grp))}


def find(directory: pd.Series, stem: str) -> int | None:
    """territory_id по названию без вида МО (src/mo/census.py: norm_name) среди МО одного региона."""
    hit = [t for t, n in directory.items() if census.norm_name(n) == census.norm_name(stem)]
    return hit[0] if len(hit) == 1 else None


def main() -> None:
    cfg = load()
    lc = cfg["lab"]
    proc, o3 = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    types = pd.read_csv(o3 / "mo_types.csv", index_col="territory_id")
    sat = pd.read_csv(o3 / "satellites.csv", index_col="territory_id")
    ids = types.index
    res = {"definition": __doc__.split("Выход")[0].strip()}

    # 1. Пригороды, названные лабораторией
    kind = mo.loc[ids, "type"]
    is_district = kind.isin(DISTRICTS)
    city_t = set(lc["city_types"])
    base = {"districts_in_city_types": round(float(types.loc[is_district, "type"].isin(city_t).mean()), 3),
            "city_okrugs_in_city_types": round(float(types.loc[kind == "городской округ", "type"].isin(city_t).mean()), 3)}
    rows = []
    for s in lc["named_suburbs"]:
        d = mo.loc[mo["region"] == s["region"], "name"]
        t, c = find(d, s["district"]), find(d, s["city"])
        r = {"district": s["district"], "region": s["region"], "city": s["city"], "in_data": t is not None,
             "city_in_data": c is not None}
        if t is not None:
            r.update(name=mo.loc[t, "name"], type=types.loc[t, "type"], city_type=types.loc[c, "type"] if c else None,
                     in_city_type=bool(types.loc[t, "type"] in city_t))
            if t in sat.index:
                g = sat.loc[t]
                r.update(group="hidden" if g["group"] == "control" and g["hidden"] else g["group"],
                         partner=g["city"], index=round(float(g["index"]), 2))
        rows.append(r)
    present = [r for r in rows if r["in_data"]]
    k, n = sum(r["in_city_type"] for r in present), len(present)
    p0 = base["districts_in_city_types"]
    res["named_suburbs"] = {"rows": rows, "n_named": len(rows), "n_in_data": n, "n_in_city_types": k, "base": base,
                            # вероятность получить не меньше k «городских» типов из n случайных районов
                            "p_binomial": round(float(sum(math.comb(n, j) * p0 ** j * (1 - p0) ** (n - j)
                                                          for j in range(k, n + 1))), 5)}
    print("пригороды лаборатории:", [(r["district"], r.get("type"), r.get("group")) for r in rows], base)

    # 2. Разложение Тейла
    w = pd.read_parquet(proc / "panel_wide.parquet")
    year, prev = lc["year"], str(int(lc["year"]) - 1)
    dates = w.index.get_level_values("date")
    spend = w.loc[dates.str.startswith(year), "Все категории"].groupby(level="territory_id").mean().loc[ids]
    spend_prev = w.loc[dates.str.startswith(prev), "Все категории"].groupby(level="territory_id").mean().loc[ids]
    lvl = features.rel_level(w)
    lvl = lvl[lvl.index.get_level_values("date").str.startswith(year)].groupby(level="territory_id").mean().loc[ids]
    K = int(types["type"].nunique())
    groups = {"types": types["type"], "regions": mo.loc[ids, "region"],
              "level_groups": pd.Series(pd.qcut(lvl, K, labels=False), index=ids)}
    pop = mo.loc[ids, "population"]
    res["theil"] = {v: {g: theil_split(spend, groups[g], pop if v == "population_weighted" else pd.Series(1.0, index=ids))
                        for g in groups} for v in ("units", "population_weighted")}
    res["theil"]["lab"] = {"clusters_between": 0.70, "regions_between": 0.15, "theil": "1,03–1,04", "n_clusters": 20}
    print("Тейл:", res["theil"])

    # 3. Рост трат: «бублики» против своих городов и обычных соседей
    growth = spend / spend_prev - 1
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    rp = rp[rp["ring"].isin(ids) & rp["city"].isin(ids)]
    gr, gc = growth.loc[rp["ring"]].to_numpy(), growth.loc[rp["city"]].to_numpy()
    ctrl = sat.index[(sat["group"] == "control") & ~sat["hidden"].astype(bool)].intersection(ids)
    hid = sat.index[(sat["group"] == "control") & sat["hidden"].astype(bool)].intersection(ids)
    res["growth"] = {
        "year": year, "median_all_pct": round(float(growth.median()) * 100, 1),
        "ring": {"n": len(rp), "median_pct": round(float(np.median(gr)) * 100, 1)},
        "ring_city": {"median_pct": round(float(np.median(gc)) * 100, 1),
                      "share_ring_faster": round(float((gr > gc).mean()), 3),
                      "median_diff_pp": round(float(np.median(gr - gc)) * 100, 1),
                      "wilcoxon_p": round(float(wilcoxon(gr, gc).pvalue), 4)},
        "hidden": {"n": len(hid), "median_pct": round(float(growth.loc[hid].median()) * 100, 1)},
        "control": {"n": len(ctrl), "median_pct": round(float(growth.loc[ctrl].median()) * 100, 1)},
        "ring_vs_control": stats.perm_test_diff(gr, growth.loc[ctrl], stat=np.median, n=10000, seed=cfg["seed"]),
        "lab": {"suburbs_cluster_growth_1q24_pct": 12.5, "suburbs_cluster_growth_1q25_pct": 1.9},
    }
    res["growth"]["ring_vs_control"]["diff"] = round(res["growth"]["ring_vs_control"]["diff"] * 100, 2)
    print("рост:", res["growth"])
    (o3 / "lab_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
