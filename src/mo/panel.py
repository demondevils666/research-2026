"""Панель трат: МО × месяц × категория, только МО с полной панелью."""

import pandas as pd

from . import data
from .config import load


def full_panel_ids(c: pd.DataFrame | None = None) -> list[int]:
    cfg = load()["panel"]
    c = data.consumption() if c is None else c
    n = c.groupby(["territory_id", "category"])["date"].nunique().unstack()
    ok = (n == cfg["months"]).all(axis=1) & (n.shape[1] == len(cfg["categories"]) + 1)
    return sorted(int(t) for t in n.index[ok])


def _to_wide(c: pd.DataFrame) -> pd.DataFrame:
    cfg = load()["panel"]
    w = c.pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="first")
    w = w[cfg["categories"] + [cfg["total_category"]]].astype(float)
    w.insert(len(cfg["categories"]), cfg["other_category"], w[cfg["total_category"]] - w[cfg["categories"]].sum(axis=1))
    w.index = w.index.set_levels(w.index.levels[0].astype(int), level=0)
    return w.sort_index()


def wide() -> pd.DataFrame:
    """Строка = (territory_id, date); колонки: траты по 5 категориям, «Прочее», итог. Руб./жителя в месяц."""
    c = data.consumption()
    ids = full_panel_ids(c)
    return _to_wide(c[c["territory_id"].isin(ids)])


def wide_partial() -> pd.DataFrame:
    """То же для МО без полной панели: только месяцы, где есть все категории (неполные месяцы отбрасываются)."""
    c = data.consumption()
    ids = full_panel_ids(c)
    return _to_wide(c[~c["territory_id"].isin(ids)]).dropna()
