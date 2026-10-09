"""Этап 4, E7: «фискальный след» — недополучают ли бюджеты пригородов налог на доходы своих жителей.

НДФЛ удерживает работодатель и перечисляет по месту своего учета (НК РФ, ст. 226, п. 7). В местный бюджет
зачисляется часть налога, взимаемого на его территории (БК РФ, ст. 61–61.6). Если жители «бублика» работают в городе,
их НДФЛ попадает в бюджет города, а траты по месту жительства остаются в районе.
Мера для пары «район — город»: разрыв = ln(траты район/город) − ln(НДФЛ на жителя район/город).
Ожидание: у «бубликов» разрыв больше, чем у обычных районов при городах (контроль).
Данные: доходы местных бюджетов по видам (Росстат БД ПМО, Y48013001, «Налог на доходы физических лиц»), 2019 год —
последний доковидный год в БД ПМО (ряды заканчиваются 2020-м); население на 1 января 2019 г.; траты СберИндекса 2023 г.
Нормативы отчислений НДФЛ различаются по видам МО и регионам. Поэтому:
  1) сравниваем «бублики» и контроль одного вида (муниципальный район против городского округа);
  2) основной тест — внутри регионов: разность средних разрывов «бубликов» и контроля в каждом регионе, где есть обе
     группы, затем среднее по регионам с бутстрап-ДИ и знаковым тестом.
Выход: outputs/stage3/fiscal_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, data, rosstat, stats  # noqa: E402
from mo.config import load, path  # noqa: E402


def budget_items(b: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fc = cfg["fiscal"]
    f = path("raw") / "rosstat" / "indicators" / f"data_{fc['budget']}_year{fc['year']}_112_v20250918.csv"
    x = pd.read_csv(f, sep=";", dtype=str)
    x = rosstat._attach(x, rosstat.oktmo_map(b, fc["year"]))
    x = x[x["indicator_period"] == fc["period"]]
    x = x[x["stdohod"].isin(fc["items"].values())]
    unit = x["indicator_unit"].iloc[0]
    k = 1000.0 if unit.startswith("Тысяча") else 1.0
    t = (x.drop_duplicates(["territory_id", "oktmo", "stdohod"])
         .pivot_table(index="territory_id", columns="stdohod", values="value", aggfunc="sum") * k)
    return t.rename(columns={v: key for key, v in fc["items"].items()}), unit


def main() -> None:
    cfg = load()
    fc, seed = cfg["fiscal"], cfg["seed"]
    nb, npm = cfg["probes"]["bootstrap"], cfg["probes"]["permutations"]
    proc, out = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    u = agglo.units(mo, rp, pd.read_parquet(proc / "edges_contiguity.parquet"))
    b = data.borders()
    bud, unit = budget_items(b, cfg)
    pop = rosstat.population_jan1(b, fc["year"])
    w = pd.read_parquet(proc / "panel_wide.parquet")
    total = cfg["panel"]["total_category"]
    spend = w[w.index.get_level_values("date").str.startswith(str(fc["spend_year"]))][total].groupby(level="territory_id").mean()
    ndfl_pc = (bud["ndfl"] / pop).where(lambda s: s > 0)
    kind = data.borders_latest(b)["municipal_district_type"]
    d = u.copy()
    d["kind_unit"], d["kind_city"] = kind.reindex(d["unit"]).to_numpy(), kind.reindex(d["partner"]).to_numpy()
    d["ls"] = np.log(spend.reindex(d["unit"]).to_numpy() / spend.reindex(d["partner"]).to_numpy())
    d["ln"] = np.log(ndfl_pc.reindex(d["unit"]).to_numpy() / ndfl_pc.reindex(d["partner"]).to_numpy())
    d["gap"] = d["ls"] - d["ln"]
    d["region"] = mo["region_code"].reindex(d["unit"]).to_numpy()
    d = d[np.isfinite(d["gap"]) & (d["kind_unit"] == fc["unit_kind"]) & (d["kind_city"] == fc["city_kind"])]
    ring, ctrl = d[d["group"] == "ring"], d[d["group"] == "control"]
    res = {"definition": __doc__.split("Выход")[0].strip(), "budget_unit": unit, "year_budget": fc["year"],
           "year_spending": fc["spend_year"], "n_ring": int(len(ring)), "n_control": int(len(ctrl)),
           "coverage_ndfl_panel": round(float(ndfl_pc.reindex(mo.index).notna().mean()), 3)}
    res["pooled"] = {
        "gap_mean_ring": round(float(ring["gap"].mean()), 3), "gap_mean_control": round(float(ctrl["gap"].mean()), 3),
        "ndfl_ratio_median_ring": round(float(np.exp(ring["ln"]).median()), 3),
        "ndfl_ratio_median_control": round(float(np.exp(ctrl["ln"]).median()), 3),
        "spend_ratio_median_ring": round(float(np.exp(ring["ls"]).median()), 3),
        "spend_ratio_median_control": round(float(np.exp(ctrl["ls"]).median()), 3),
        "perm_gap_ring_minus_control": stats.perm_test_diff(ring["gap"], ctrl["gap"], n=npm, seed=seed),
        "gap_ci95_ring": [round(v, 3) for v in stats.bootstrap_ci(ring["gap"], n=nb, seed=seed)],
        "gap_ci95_control": [round(v, 3) for v in stats.bootstrap_ci(ctrl["gap"], n=nb, seed=seed)],
    }
    # Внутри регионов: одинаковые региональные нормативы для «бублика» и контроля
    by = []
    for r, g in d.groupby("region"):
        a, c = g[g["group"] == "ring"]["gap"], g[g["group"] == "control"]["gap"]
        if len(a) and len(c):
            by.append(float(a.mean() - c.mean()))
    by = np.array(by)
    res["within_region"] = {
        "n_regions": int(len(by)), "mean_diff": round(float(by.mean()), 3) if len(by) else None,
        "ci95": [round(v, 3) for v in stats.bootstrap_ci(by, n=nb, seed=seed)] if len(by) > 2 else None,
        "share_regions_ring_gt_control": round(float((by > 0).mean()), 3) if len(by) else None,
        "sign_test_p": float(binomtest(int((by > 0).sum()), len(by), 0.5).pvalue) if len(by) else None,
    }
    res["interpretation"] = ("Разрыв > 0: жители тратят относительно города больше, чем их местный бюджет получает НДФЛ "
                             "относительно города. Если у «бубликов» разрыв больше, чем у обычных районов, часть НДФЛ их "
                             "жителей уходит в бюджет города (по месту работы).")
    (out / "fiscal_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
