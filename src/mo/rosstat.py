"""Росстат (БД ПМО, «Если быть точным»): население, зарплата, структура занятости по МО.

Сопоставление с territory_id идет через ВСЕ версии справочника СберИндекса: ОКТМО, действовавший в год
наблюдения, ищем среди ОКТМО всех версий; если не нашли — пробуем oktmo_stable Росстата.
"""

import pandas as pd

from . import data
from .config import load

TOP = "Муниципальное образование верхнего уровня"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", ".", regex=False).str.replace(" ", "", regex=False), errors="coerce")


def oktmo_map(b: pd.DataFrame, year: int) -> dict:
    """ОКТМО (8 знаков) -> territory_id для года наблюдения.

    При смене границ СберИндекс выдает новый territory_id с тем же ОКТМО, поэтому сначала берем версию,
    действовавшую в этом году (year_from <= year < year_to), а для остальных кодов — любую версию,
    если код однозначен.
    """
    m = pd.DataFrame({"ok": data.oktmo8(b["oktmo"]), "tid": b["territory_id"].astype(int),
                      "active": (b["year_from"] <= year) & (year < b["year_to"])})
    act = m[m["active"]].drop_duplicates(["ok", "tid"])
    act = act[~act["ok"].duplicated(keep=False)]
    rest = m[~m["ok"].isin(act["ok"])].drop_duplicates(["ok", "tid"])
    rest = rest[~rest["ok"].duplicated(keep=False)]
    return {**dict(zip(rest["ok"], rest["tid"])), **dict(zip(act["ok"], act["tid"]))}


def _attach(x: pd.DataFrame, omap: dict) -> pd.DataFrame:
    x = x[x["mun_level"] == TOP].copy()
    tid = x["oktmo"].map(omap)
    stable = x["oktmo_stable"].where(x["oktmo_stable"].str.len() == 8)
    x["territory_id"] = tid.fillna(stable.map(omap))
    x["value"] = _num(x["indicator_value"])
    return x.dropna(subset=["territory_id", "value"]).astype({"territory_id": int})


def population(b: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    """Население на 1 января и доля горожан за год с санити-правилами."""
    cfg = load()["rosstat"]
    diag: dict = {}
    pop = _attach(data.rosstat_indicator(cfg["population"], year), oktmo_map(b, year))
    # Обычно «На 1 января»; у части регионов (например, Кемеровская область) — «Значение показателя за год»
    pop["prio"] = (pop["indicator_period"] != "На 1 января").astype(int)
    pop = pop.sort_values("prio").drop_duplicates(["territory_id", "mest", "oktmo"])
    pop = pop[pop["prio"] == pop.groupby("territory_id")["prio"].transform("min")]
    diag["period_counts"] = pop.drop_duplicates("territory_id")["indicator_period"].value_counts().to_dict()
    p = pop.pivot_table(index="territory_id", columns="mest", values="value", aggfunc="sum")
    total = p.get("Все население")
    urban = p.get("Городское население").reindex(total.index).fillna(0)
    rural = p.get("Сельское население").reindex(total.index).fillna(0)
    # Санити: итог должен быть > 0 и не меньше частей, а части в сумме ≈ итог.
    # Встречаются ошибки источника (Краснинский р-н Липецкой обл. 2023: итог 0, город = село = 13 303).
    bad_total = (total <= 0) | (total < urban) | (total < rural)
    pop_ok = total.where(~bad_total, pd.concat([urban, rural], axis=1).max(axis=1))
    consistent = ((urban + rural) - pop_ok).abs() <= 0.02 * pop_ok
    out = pd.DataFrame({"population": pop_ok, "urban_share": (urban / pop_ok).clip(0, 1).where(consistent)})
    diag["fixed_total"] = [int(t) for t in pop_ok.index[bad_total]]
    diag["urban_share_undefined"] = [int(t) for t in pop_ok.index[~consistent]]
    return out, diag


def population_jan1(b: pd.DataFrame, year: int) -> pd.Series:
    """Все население на 1 января года (только этот период, без подстановок; итог > 0). Для расчета роста."""
    x = _attach(data.rosstat_indicator(load()["rosstat"]["population"], year), oktmo_map(b, year))
    x = x[(x["indicator_period"] == "На 1 января") & (x["mest"] == "Все население")]
    s = x.drop_duplicates(["territory_id", "oktmo"]).groupby("territory_id")["value"].sum()
    return s[s > 0]


def context(b: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Таблица контекста Росстата на год cfg.rosstat.year + диагностика сопоставления."""
    cfg = load()["rosstat"]
    year = cfg["year"]
    omap = oktmo_map(b, year)
    diag = {"year": year}

    out, d = population(b, year)
    diag["population"] = {k: (v if not isinstance(v, list) else len(v)) for k, v in d.items()}
    diag["population_fixed_ids"] = d["fixed_total"]
    # Доля горожан, если в основном году разбивка несогласована, — из соседних лет
    for y in cfg.get("urban_share_fallback_years", []):
        miss = out["urban_share"].isna()
        if not miss.any():
            break
        alt, _ = population(b, y)
        out.loc[miss, "urban_share"] = alt["urban_share"].reindex(out.index[miss])
    diag["urban_share_still_undefined"] = [int(t) for t in out.index[out["urban_share"].isna()]]

    wag = _attach(data.rosstat_indicator(cfg["wages"], year), omap)
    wag = wag[(wag["indicator_period"] == cfg["period_annual"]) & (wag["okved2"] == cfg["total_okved"])]
    out["wage"] = wag.groupby("territory_id")["value"].mean()

    emp = _attach(data.rosstat_indicator(cfg["employment"], year), omap)
    emp = emp[emp["indicator_period"] == cfg["period_annual"]]
    total = emp[emp["okved2"] == cfg["total_okved"]].groupby("territory_id")["value"].sum()
    out["employees"] = total
    sect = emp[emp["okved2"].str.startswith("Раздел")].copy()
    sect["letter"] = sect["okved2"].str.split().str[1]
    letter2block = {l: blk for blk, ls in cfg["okved_blocks"].items() for l in ls}
    sect["block"] = sect["letter"].map(letter2block)
    diag["okved_letters_unmapped"] = sorted(sect.loc[sect["block"].isna(), "letter"].unique().tolist())
    blocks = sect.dropna(subset=["block"]).pivot_table(index="territory_id", columns="block", values="value", aggfunc="sum")
    for blk in cfg["okved_blocks"]:
        out[f"emp_{blk}"] = (blocks.get(blk, pd.Series(dtype=float)).reindex(out.index).fillna(0) / total.reindex(out.index))
    # Доля занятости, которую покрывают видимые разделы (Росстат скрывает конфиденциальные ячейки)
    out["emp_covered"] = blocks.sum(axis=1).reindex(out.index) / total.reindex(out.index)
    diag["n_mo"] = int(out["population"].notna().sum())
    return out, diag
