"""Тесты мер для пар «район — город» (src/mo/agglo.py) на игрушечных данных."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo  # noqa: E402


def _toy():
    rng = np.random.default_rng(0)
    T = 23
    city = rng.normal(size=T)
    rows = {1: city, 2: city + 0.1 * rng.normal(size=T)}  # 1 — город, 2 — «бублик», синхронный с городом
    for t in range(3, 9):  # 3..8 — обычные районы региона, свой общий фон
        rows[t] = rng.normal(size=T)
    dl = pd.DataFrame(rows).T
    region = pd.Series(10, index=dl.index)
    u = pd.DataFrame({"group": ["ring", "control"], "unit": [2, 3], "partner": [1, 1], "road_km": [0.0, 20.0]})
    return dl, region, u


def test_comovement_excess_detects_city_sync():
    dl, region, u = _toy()
    cm = agglo.comovement(dl, region, u, pool=set(range(3, 9)), min_obs=11).set_index("unit")
    assert cm.loc[2, "corr_city"] > 0.9
    assert cm.loc[2, "excess"] > 0.7
    assert abs(cm.loc[3, "excess"]) < 0.6
    # Пропуски: корреляция по общим месяцам, при нехватке наблюдений — NaN
    dl2 = dl.copy()
    dl2.iloc[1, :15] = np.nan
    cm2 = agglo.comovement(dl2, region, u, pool=set(range(3, 9)), min_obs=11).set_index("unit")
    assert np.isnan(cm2.loc[2, "corr_city"])


def test_wage_residual_and_index():
    u = pd.DataFrame({"group": ["control"] * 4 + ["ring"], "unit": [11, 12, 13, 14, 15], "partner": [1] * 5})
    wage = pd.Series({1: 100.0, 11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 50.0})
    spend = pd.Series({1: 100.0, 11: 50.0, 12: 60.0, 13: 70.0, 14: 80.0, 15: 95.0})
    wr = agglo.wage_residual(spend, wage, u).set_index("unit")
    # Контроль лежит на прямой: остатки ~0; «бублик» тратит почти как город при низкой зарплате: остаток > 0
    assert wr.loc[[11, 12, 13, 14], "resid"].abs().max() < 1e-9
    assert wr.loc[15, "resid"] > 0.5
    sig = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, 2.0, 3.0]})
    idx = agglo.satellite_index(sig, ["a", "b"], min_signals=2)
    assert idx.iloc[0] < idx.iloc[1] and np.isnan(idx.iloc[2])


def test_units_control_only_city_of_own_region(monkeypatch):
    # 1, 2 — городские округа регионов 10 и 20; 3 — район региона 10, граничит с обоими городами;
    # 4 — район региона 10, граничит только с городом чужого региона; 5 — «бублик» вокруг города 1
    mo = pd.DataFrame({"type": ["городской округ"] * 2 + ["муниципальный район"] * 3, "shape": [1, 1, 1, 1, 3],
                       "region_code": [10, 20, 10, 10, 10]}, index=[1, 2, 3, 4, 5])
    rp = pd.DataFrame({"ring": [5], "city": [1]})
    cont = pd.DataFrame({"a": [3, 3, 4, 5], "b": [1, 2, 2, 1]})
    # до города чужого региона дорога короче: без ограничения взяли бы его
    con = pd.DataFrame({"type": "highway", "territory_id_x": [3, 3, 4], "territory_id_y": [1, 2, 2], "distance": [50.0, 10.0, 5.0]})
    monkeypatch.setattr(agglo.data, "hack", lambda name: con)
    u = agglo.units(mo, rp, cont, same_region=True).set_index("unit")
    assert u.loc[3, "partner"] == 1 and 4 not in u.index and u.loc[5, "group"] == "ring"
    u_any = agglo.units(mo, rp, cont, same_region=False).set_index("unit")
    assert u_any.loc[3, "partner"] == 2 and u_any.loc[4, "partner"] == 2
