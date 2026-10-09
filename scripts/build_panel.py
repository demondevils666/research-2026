"""Этап 1: панель, признаки, таблица МО с контекстом, физические связи.

Выход (data/processed/, в git не идет):
  panel_wide.parquet      — (territory_id, date) × траты по категориям, «Прочее», итог
  features_monthly.parquet — CLR долей + относительный уровень по месяцам (не стандартизованы)
  profile.parquet         — годовой профиль МО (год из configs/base.yaml)
  mo.parquet              — атрибуты МО: название, регион, тип, shape, партнер-«бублик», Росстат, доступность рынков
  edges_contiguity.parquet, edges_road_knn.parquet, ring_pairs.parquet
Сводка с проверками: outputs/panel/summary.json (в git).
Запуск: python scripts/build_panel.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import data, features, links, panel, rosstat  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "panel"
    proc.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    s: dict = {}

    w = panel.wide()
    ids = sorted(w.index.get_level_values("territory_id").unique().tolist())
    s["n_mo"], s["n_months"] = len(ids), int(w.index.get_level_values("date").nunique())
    s["rows"] = len(w)
    other = w[cfg["panel"]["other_category"]]
    s["other_negative_rows"] = int((other < 0).sum())
    s["other_share_median"] = round(float((other / w[cfg["panel"]["total_category"]]).median()), 4)

    fm = features.monthly(w)
    pr = features.profile(w)
    sh = features.shares(w)
    s["shares_sum_max_abs_err"] = float((sh.sum(axis=1) - 1).abs().max())
    clr_cols = [c for c in fm.columns if c.startswith("clr_")]
    s["clr_rowsum_max_abs"] = float(fm[clr_cols].sum(axis=1).abs().max())
    s["features_nan"] = int(fm.isna().sum().sum() + pr.isna().sum().sum())
    s["profile_year"] = cfg["features"]["profile_year"]

    # Атрибуты МО
    b = data.borders()
    latest = data.borders_latest(b).loc[ids]
    mo = latest[["municipal_district_name", "municipal_district_type", "municipal_district_status", "region_code",
                 "region_name", "shape", "municipal_district_center_lat", "municipal_district_center_lon", "year_to"]].copy()
    mo.columns = ["name", "type", "status", "region_code", "region", "shape", "lat", "lon", "version_year_to"]
    rp_all = links.ring_pairs(b[b["year_to"] == 9999])
    rp = rp_all[rp_all["ring"].isin(ids) & rp_all["city"].isin(ids)]
    mo["ring_city"] = pd.Series(dict(zip(rp["ring"], rp["city"]))).reindex(mo.index)
    mo["is_ring_city"] = mo.index.isin(rp["city"])
    ma = data.hack("market_access").set_index("territory_id")["market_access"]
    mo["market_access"] = ma.reindex(mo.index)
    ctx, diag = rosstat.context(b)
    mo = mo.join(ctx, how="left")
    s["rosstat"] = diag
    s["rosstat_coverage"] = {c: int(mo[c].notna().sum()) for c in ["population", "urban_share", "wage", "employees"]}
    s["rosstat"]["urban_share_still_undefined"] = [t for t in s["rosstat"]["urban_share_still_undefined"] if t in mo.index]
    s["rosstat_coverage_share"] = round(float(mo["population"].notna().mean()), 4)
    s["emp_covered_median"] = round(float(mo["emp_covered"].median()), 3)
    s["market_access_missing"] = int(mo["market_access"].isna().sum())
    s["mo_types"] = mo["type"].value_counts().to_dict()
    s["mo_not_active_version"] = int((mo["version_year_to"] != 9999).sum())

    # Связи
    cont = links.contiguity(set(ids))
    road = links.road_knn(set(ids), cfg["links"]["road_knn"])
    deg = pd.concat([cont["a"], cont["b"]]).value_counts().reindex(ids).fillna(0)
    s["contiguity"] = {"pairs": len(cont), "isolated": int((deg == 0).sum()), "degree_median": float(deg.median())}
    s["road_knn"] = {"k": cfg["links"]["road_knn"], "pairs": len(road), "km_median": float(road["km"].median())}
    s["ring_pairs_total_active"] = len(rp_all)
    s["ring_pairs_in_panel"] = len(rp)
    # Проверка ориентации пар: «бублик» должен быть районом или округом, центр — городским округом
    s["ring_pairs_ring_types"] = mo.loc[rp["ring"], "type"].value_counts().to_dict()
    s["ring_pairs_city_types"] = mo.loc[rp["city"], "type"].value_counts().to_dict()
    s["ring_pairs_city_is_region_capital"] = int(mo.loc[rp["city"], "status"].astype(str).str.contains("центр_субъекта").sum())

    w.to_parquet(proc / "panel_wide.parquet")
    fm.to_parquet(proc / "features_monthly.parquet")
    pr.to_parquet(proc / "profile.parquet")
    mo.to_parquet(proc / "mo.parquet")
    cont.to_parquet(proc / "edges_contiguity.parquet")
    road.to_parquet(proc / "edges_road_knn.parquet")
    rp.to_parquet(proc / "ring_pairs.parquet")
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
