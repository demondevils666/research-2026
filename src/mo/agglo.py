"""Агломерационные пары: «бублики» (район вокруг города с общим центром) и контрольная группа.

Контроль — обычные районы и муниципальные округа (shape = 1), граничащие с городским округом своего региона
(configs/base.yaml: agglo.control_same_region); если таких городов несколько, берется ближайший по дороге.
Город другого региона не годится: близость района к городу сравнивается с районами его региона, и чужой регион
занижает ее сам по себе.
"""

import numpy as np
import pandas as pd

from . import data
from .config import load

RURAL = {"муниципальный район", "муниципальный округ"}


def units(mo: pd.DataFrame, rp: pd.DataFrame, cont: pd.DataFrame, same_region: bool | None = None) -> pd.DataFrame:
    """Таблица единиц сравнения: group (ring/control), unit (район), partner (город), road_km.
    same_region — контроль только с городом своего региона (по умолчанию из конфига)."""
    if same_region is None:
        same_region = load()["agglo"]["control_same_region"]
    rings = pd.DataFrame({"group": "ring", "unit": rp["ring"], "partner": rp["city"], "road_km": 0.0})
    ring_ids = set(rp["ring"])
    city_ok = set(mo.index[mo["type"] == "городской округ"])
    plain = set(mo.index[mo["type"].isin(RURAL) & (mo["shape"] == 1)]) - ring_ids
    e = pd.concat([cont.rename(columns={"a": "u", "b": "v"}), cont.rename(columns={"a": "v", "b": "u"})])
    e = e[e["u"].isin(plain) & e["v"].isin(city_ok)]
    if same_region:
        reg = mo["region_code"]
        e = e[reg.reindex(e["u"]).to_numpy() == reg.reindex(e["v"]).to_numpy()]
    con = data.hack("connection")
    con = con[con["type"] == "highway"]
    km = {(min(x, y), max(x, y)): d for x, y, d in zip(con["territory_id_x"], con["territory_id_y"], con["distance"])}
    e = e.assign(road_km=[km.get((min(u, v), max(u, v)), np.nan) for u, v in zip(e["u"], e["v"])])
    e = e.sort_values("road_km").drop_duplicates("u")
    ctrl = pd.DataFrame({"group": "control", "unit": e["u"], "partner": e["v"], "road_km": e["road_km"]})
    return pd.concat([rings, ctrl], ignore_index=True)


def coherence(labels: pd.Series, u: pd.DataFrame) -> dict:
    """Связность агломераций: доля пар, где район и его город в одном типе (ring против control)."""
    same = labels.loc[u["unit"]].to_numpy() == labels.loc[u["partner"]].to_numpy()
    out = {g: float(same[(u["group"] == g).to_numpy()].mean()) for g in ("ring", "control")}
    out["gap"] = out["ring"] - out["control"]
    return out


def peer_pool(mo: pd.DataFrame, rp: pd.DataFrame) -> set:
    """«Обычные» районы и округа (shape = 1, не «бублики»): с ними сравниваем район, а не со всеми МО."""
    return set(mo.index[mo["type"].isin(RURAL) & (mo["shape"] == 1)]) - set(rp["ring"])


def _peers(region: pd.Series, pool: set) -> dict:
    return {r: [t for t in g.index if t in pool] for r, g in region.groupby(region)}


def profile_closeness(X: pd.DataFrame, region: pd.Series, u: pd.DataFrame, pool: set) -> pd.DataFrame:
    """Близость профиля района к своему городу против «обычных» районов своего региона (проба P1).

    d_city = ||x_r − x_c||, d_peer = медиана ||x_r − x_p|| по районам p из pool того же региона (без r и c);
    closer = d_city < d_peer, ratio = d_city / d_peer. X — стандартизованные признаки (строки — territory_id),
    region — код региона для всех МО из X. Пары, где у региона меньше трех «сверстников», пропускаются.
    """
    rows = []
    Xv = X.to_numpy()
    pos = {t: i for i, t in enumerate(X.index)}
    by_region = _peers(region.loc[[t for t in region.index if t in pool]], pool)
    for _, r in u.iterrows():
        peers = [p for p in by_region.get(region[r["unit"]], []) if p not in (r["unit"], r["partner"])]
        if len(peers) < 3:
            continue
        xu = Xv[pos[r["unit"]]]
        d_city = float(np.linalg.norm(xu - Xv[pos[r["partner"]]]))
        d_peer = float(np.median(np.linalg.norm(Xv[[pos[p] for p in peers]] - xu, axis=1)))
        rows.append({"group": r["group"], "unit": int(r["unit"]), "partner": int(r["partner"]),
                     "d_city": d_city, "d_peer": d_peer, "closer": d_city < d_peer, "ratio": d_city / d_peer,
                     "road_km": r["road_km"]})
    return pd.DataFrame(rows)


def _corr(a: np.ndarray, b: np.ndarray, min_obs: int) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_obs or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def comovement(dl: pd.DataFrame, region: pd.Series, u: pd.DataFrame, pool: set, min_obs: int = 11) -> pd.DataFrame:
    """Синхронность помесячных изменений трат района с городом и с «обычными» районами своего региона.

    dl — помесячные изменения относительного уровня трат (строки — territory_id, пропуски допустимы:
    корреляция считается по общим месяцам, не меньше min_obs). corr_city — корреляция с городом,
    corr_peer — медиана корреляций с районами своего региона, excess = corr_city − corr_peer
    (синхронность именно с городом, сверх общего регионального фона).
    """
    D = dl.to_numpy()
    pos = {t: i for i, t in enumerate(dl.index)}
    by_region = _peers(region.loc[[t for t in region.index if t in pool]], pool)
    rows = []
    for _, r in u.iterrows():
        a = D[pos[r["unit"]]]
        c_city = _corr(a, D[pos[r["partner"]]], min_obs)
        peers = [p for p in by_region.get(region[r["unit"]], []) if p not in (r["unit"], r["partner"]) and p in pos]
        cp = [_corr(a, D[pos[p]], min_obs) for p in peers]
        cp = [v for v in cp if np.isfinite(v)]
        c_peer = float(np.median(cp)) if len(cp) >= 3 else np.nan
        rows.append({"group": r["group"], "unit": int(r["unit"]), "partner": int(r["partner"]),
                     "corr_city": c_city, "corr_peer": c_peer, "excess": c_city - c_peer,
                     "n_obs": int((np.isfinite(a) & np.isfinite(D[pos[r["partner"]]])).sum())})
    return pd.DataFrame(rows)


def wage_residual(spend: pd.Series, wage: pd.Series, u: pd.DataFrame) -> pd.DataFrame:
    """«Место работы против места жительства» для каждой пары.

    ls = ln(траты района / траты города), lw = ln(зарплата района / зарплата города) по Росстату (место работы).
    У обычных районов траты следуют за местной зарплатой: регрессия ls = a + b·lw оценивается на контроле.
    resid = ls − (a + b·lw) > 0: жители тратят больше, чем объясняет местная зарплата (зарабатывают в другом месте).
    """
    ls = np.log(spend.reindex(u["unit"]).to_numpy() / spend.reindex(u["partner"]).to_numpy())
    lw = np.log(wage.reindex(u["unit"]).to_numpy() / wage.reindex(u["partner"]).to_numpy())
    ctrl = (u["group"] == "control").to_numpy() & np.isfinite(ls) & np.isfinite(lw)
    b, a = np.polyfit(lw[ctrl], ls[ctrl], 1)
    return pd.DataFrame({"group": u["group"].to_numpy(), "unit": u["unit"].to_numpy(), "partner": u["partner"].to_numpy(),
                         "ls": ls, "lw": lw, "resid": ls - (a + b * lw)}).assign(fit_a=a, fit_b=b)


def satellite_index(signals: pd.DataFrame, cols: list[str], min_signals: int = 2) -> pd.Series:
    """Индекс «спутника»: среднее z-оценок сигналов (знак: больше = больше похож на город).

    z считаются по всем парам вместе; индекс определен, если доступно не меньше min_signals сигналов.
    """
    Z = (signals[cols] - signals[cols].mean()) / signals[cols].std(ddof=0)
    return Z.mean(axis=1).where(Z.notna().sum(axis=1) >= min_signals)
