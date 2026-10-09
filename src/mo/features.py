"""Признаки МО: структура трат (CLR долей) и относительный уровень.

Доли шести категорий (пять названных + «Прочее») — композиционные данные: в сумме 1, поэтому сравниваем их
в CLR-координатах Айчисона: clr_j = ln(доля_j) − среднее ln(долей).
Относительный уровень = ln(траты на жителя) − медиана ln по всем МО в этом месяце: снимает инфляцию и общую
сезонность, остается положение МО относительно страны.
"""

import numpy as np
import pandas as pd

from .config import load


def share_cols() -> list[str]:
    cfg = load()["panel"]
    return cfg["categories"] + [cfg["other_category"]]


def shares(w: pd.DataFrame) -> pd.DataFrame:
    cfg = load()["panel"]
    return w[share_cols()].div(w[cfg["total_category"]], axis=0)


def clr(sh: pd.DataFrame) -> pd.DataFrame:
    lg = np.log(sh.clip(lower=load()["panel"]["share_floor"]))
    out = lg.sub(lg.mean(axis=1), axis=0)
    out.columns = [f"clr_{c}" for c in sh.columns]
    return out


def rel_level(w: pd.DataFrame) -> pd.Series:
    lg = np.log(w[load()["panel"]["total_category"]])
    return (lg - lg.groupby(level="date").transform("median")).rename("rel_level")


def monthly(w: pd.DataFrame) -> pd.DataFrame:
    """Помесячные признаки (не стандартизованы): CLR долей + относительный уровень."""
    return pd.concat([clr(shares(w)), rel_level(w)], axis=1)


def profile(w: pd.DataFrame, year: str | None = None) -> pd.DataFrame:
    """Годовой профиль МО: CLR средних за год долей + средний относительный уровень."""
    year = year or load()["features"]["profile_year"]
    wy = w[w.index.get_level_values("date").str.startswith(year)]
    mean_sh = shares(wy).groupby(level="territory_id").mean()
    lvl = rel_level(w).loc[wy.index].groupby(level="territory_id").mean()
    return pd.concat([clr(mean_sh), lvl], axis=1)


def median_log_level(w: pd.DataFrame) -> pd.Series:
    """Медиана ln(траты на жителя) по МО в каждом месяце — база относительного уровня."""
    return np.log(w[load()["panel"]["total_category"]]).groupby(level="date").median()


def profile_partial(w: pd.DataFrame, median_log: pd.Series, year: str | None = None) -> pd.DataFrame:
    """Годовой профиль по доступным месяцам года для МО вне панели: CLR средних долей + средний уровень относительно
    медианы панели в тех же месяцах (median_log). Для МО панели и median_log = median_log_level(панель) равен profile()."""
    year = year or load()["features"]["profile_year"]
    wy = w[w.index.get_level_values("date").str.startswith(year)]
    mean_sh = shares(wy).groupby(level="territory_id").mean()
    lg = np.log(wy[load()["panel"]["total_category"]])
    lvl = (lg - median_log.reindex(wy.index.get_level_values("date")).to_numpy()).groupby(level="territory_id").mean()
    return pd.concat([clr(mean_sh), lvl.rename("rel_level")], axis=1)


def standardize(x: pd.DataFrame) -> pd.DataFrame:
    return (x - x.mean()) / x.std(ddof=0)
