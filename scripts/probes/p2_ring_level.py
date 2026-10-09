"""Проба P2 (этап 1): чем «бублик» отличается от своего города по уровню и структуре трат.

Для пар «бублик — его город» и контрольных пар «обычный район — соседний городской округ» (как в P1)
считаем log-отношения район/город за 2024 год: траты на жителя всего и по категориям, доли категорий,
а также зарплату Росстата (2023) и разницу в доле горожан. Вопрос: «бублик» похож на город, потому что
его жители живут как горожане (доход, урбанизация), или только по структуре трат?
Выход: outputs/probes/p2_ring_level.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mo import features, stats  # noqa: E402
from mo.config import load, path  # noqa: E402
from mo.agglo import units  # noqa: E402


def q3(s: pd.Series) -> list[float]:
    return [round(float(s.quantile(p)), 3) for p in (0.25, 0.5, 0.75)]


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "probes"
    mo = pd.read_parquet(proc / "mo.parquet")
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    cont = pd.read_parquet(proc / "edges_contiguity.parquet")
    w = pd.read_parquet(proc / "panel_wide.parquet")
    year = cfg["features"]["profile_year"]
    wy = w[w.index.get_level_values("date").str.startswith(year)].groupby(level="territory_id").mean()
    sh = features.shares(wy)
    u = units(mo, rp, cont)

    res = {"definition": __doc__.split("Выход")[0].strip(), "year_spending": year, "year_rosstat": cfg["rosstat"]["year"]}
    cats = features.share_cols() + [cfg["panel"]["total_category"]]
    for grp in ("ring", "control"):
        g = u[u["group"] == grp]
        a, c = g["unit"].to_numpy(), g["partner"].to_numpy()
        r = {"n": len(g)}
        r["level_ratio_unit_to_city_q25_q50_q75"] = q3(pd.Series(wy.loc[a, cfg["panel"]["total_category"]].to_numpy()
                                                               / wy.loc[c, cfg["panel"]["total_category"]].to_numpy()))
        r["share_unit_below_city_level"] = round(float((wy.loc[a, cfg["panel"]["total_category"]].to_numpy()
                                                        < wy.loc[c, cfg["panel"]["total_category"]].to_numpy()).mean()), 3)
        r["per_capita_ratio_by_category_median"] = {k: round(float(np.median(wy.loc[a, k].to_numpy() / wy.loc[c, k].to_numpy())), 3) for k in cats}
        r["share_diff_pp_unit_minus_city_median"] = {k: round(float(np.median(sh.loc[a, k].to_numpy() - sh.loc[c, k].to_numpy()) * 100), 2)
                                                     for k in features.share_cols()}
        wage_ratio = pd.Series(mo.loc[a, "wage"].to_numpy() / mo.loc[c, "wage"].to_numpy())
        r["wage_ratio_unit_to_city_q25_q50_q75"] = q3(wage_ratio)
        urb = pd.Series(mo.loc[a, "urban_share"].to_numpy() - mo.loc[c, "urban_share"].to_numpy())
        r["urban_share_diff_unit_minus_city_q25_q50_q75"] = q3(urb.dropna())
        r["unit_urban_share_q25_q50_q75"] = q3(mo.loc[a, "urban_share"].dropna())
        lvl = np.log(wy.loc[a, cfg["panel"]["total_category"]].to_numpy() / wy.loc[c, cfg["panel"]["total_category"]].to_numpy())
        ok = np.isfinite(np.log(wage_ratio.to_numpy()))
        r["corr_log_level_ratio_vs_log_wage_ratio"] = round(float(np.corrcoef(lvl[ok], np.log(wage_ratio.to_numpy()[ok]))[0, 1]), 3)
        res[grp] = r
    # Разница групп по относительному уровню трат (район/город): перестановочный тест медиан
    lv = {}
    for grp in ("ring", "control"):
        g = u[u["group"] == grp]
        lv[grp] = np.log(wy.loc[g["unit"], cfg["panel"]["total_category"]].to_numpy()
                         / wy.loc[g["partner"], cfg["panel"]["total_category"]].to_numpy())
    res["perm_median_log_level_ratio_ring_minus_control"] = stats.perm_test_diff(
        lv["ring"], lv["control"], stat=np.median, n=cfg["probes"]["permutations"], seed=cfg["seed"])
    (out / "p2_ring_level.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
