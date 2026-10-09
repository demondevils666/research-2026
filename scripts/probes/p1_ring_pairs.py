"""Проба P1 (этап 1): «бублики» против правильного контроля.

Вопрос. Район-«бублик» (shape = 3 в данных: его центр лежит в соседнем городе, см. src/mo/links.py) по тратам
ближе к своему городу, чем к типичному району своего региона. Это особенность «бубликов» или так выглядит любой район, граничащий с городом?

Мера для единицы (район r, город c):
  d_city = ||x_r − x_c||, d_peer = медиана ||x_r − x_p|| по «обычным» районам p своего региона
  (муниципальные районы и округа, shape=1, не «бублики», не r и не c); closer = d_city < d_peer; ratio = d_city / d_peer.
Группы: «бублики» (65 пар в полной панели) против контроля = обычные районы (shape=1), имеющие общую границу
с городским округом своего региона (если таких городов несколько — берется ближайший по дороге).
Чувствительность: контроль из всех соседних городов, включая города других регионов.
x — годовой профиль 2024 (CLR шести долей + относительный уровень), стандартизованный по 2 016 МО.
Проверки устойчивости: только структура (без уровня); отдельно для пар, где центр — столица субъекта и
где нет (контроль делится так же по статусу соседнего города); помесячно.
Выход: outputs/probes/p1_ring_pairs.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mo import features, stats  # noqa: E402
from mo.agglo import RURAL, profile_closeness, units  # noqa: E402
from mo.config import load, path  # noqa: E402


def measure(X: pd.DataFrame, mo: pd.DataFrame, u: pd.DataFrame, peer_pool: set) -> pd.DataFrame:
    return profile_closeness(X, mo["region_code"], u, peer_pool)


def compare(m: pd.DataFrame, cfg: dict) -> dict:
    a, b = m[m["group"] == "ring"], m[m["group"] == "control"]
    seed, nb, npm = cfg["seed"], cfg["probes"]["bootstrap"], cfg["probes"]["permutations"]
    return {
        "n_ring": len(a), "n_control": len(b),
        "share_closer_ring": round(float(a["closer"].mean()), 3),
        "share_closer_ring_ci95": [round(v, 3) for v in stats.bootstrap_ci(a["closer"], n=nb, seed=seed)],
        "share_closer_control": round(float(b["closer"].mean()), 3),
        "share_closer_control_ci95": [round(v, 3) for v in stats.bootstrap_ci(b["closer"], n=nb, seed=seed)],
        "perm_share_diff": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in stats.perm_test_diff(a["closer"], b["closer"], n=npm, seed=seed).items()},
        "median_ratio_ring": round(float(a["ratio"].median()), 3),
        "median_ratio_control": round(float(b["ratio"].median()), 3),
        "perm_median_ratio_diff": {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in stats.perm_test_diff(a["ratio"], b["ratio"], stat=np.median, n=npm, seed=seed).items()},
        "control_road_km_median": round(float(b["road_km"].median()), 1),
    }


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "probes"
    out.mkdir(parents=True, exist_ok=True)
    mo = pd.read_parquet(proc / "mo.parquet")
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    cont = pd.read_parquet(proc / "edges_contiguity.parquet")
    prof = pd.read_parquet(proc / "profile.parquet")
    X = features.standardize(prof)
    u = units(mo, rp, cont)
    ring_ids = set(rp["ring"])
    peer_pool = set(mo.index[mo["type"].isin(RURAL) & (mo["shape"] == 1)]) - ring_ids
    res = {"definition": __doc__.split("Выход")[0].strip()}

    # Повтор пробы этапа 0 (пиры: все районы и округа региона, кроме «бубликов»), только «бублики»
    pool0 = set(mo.index[mo["type"].isin(RURAL)]) - ring_ids
    m0 = measure(X, mo, u[u["group"] == "ring"], pool0)
    res["stage0_replication_share_closer"] = round(float(m0["closer"].mean()), 3)

    m = measure(X, mo, u, peer_pool)
    res["main"] = compare(m, cfg)
    clr_cols = [c for c in X.columns if c.startswith("clr_")]
    res["structure_only"] = compare(measure(features.standardize(prof[clr_cols]), mo, u, peer_pool), cfg)
    cap = u["partner"].map(mo["status"].astype(str).str.contains("центр_субъекта"))
    res["partner_is_region_capital"] = compare(measure(X, mo, u[cap], peer_pool), cfg)
    res["partner_not_region_capital"] = compare(measure(X, mo, u[~cap], peer_pool), cfg)
    res["control_any_region"] = compare(measure(X, mo, units(mo, rp, cont, same_region=False), peer_pool), cfg)

    # Помесячно: тот же расчет на стандартизованных месячных признаках
    fm = features.standardize(pd.read_parquet(proc / "features_monthly.parquet"))
    monthly = {}
    for date, g in fm.groupby(level="date"):
        mm = measure(g.droplevel("date"), mo, u, peer_pool)
        monthly[date] = [round(float(mm.loc[mm["group"] == k, "closer"].mean()), 3) for k in ("ring", "control")]
    res["monthly_share_closer_ring_control"] = monthly
    vals = np.array(list(monthly.values()))
    res["monthly_ring_gt_control_months"] = int((vals[:, 0] > vals[:, 1]).sum())

    # Крайние случаи для ручного просмотра
    top = m[m["group"] == "ring"].sort_values("ratio")
    names = mo["name"]
    res["ring_most_city_like"] = [f"{names[r.unit]} → {names[r.partner]} ({r.ratio:.2f})" for r in top.head(5).itertuples()]
    res["ring_least_city_like"] = [f"{names[r.unit]} → {names[r.partner]} ({r.ratio:.2f})" for r in top.tail(5).itertuples()]
    m.to_csv(out / "p1_units.csv", index=False)
    (out / "p1_ring_pairs.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("definition", "monthly_share_closer_ring_control")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
