"""Инвентаризация исходных данных: схемы, охват, пропуски, связи МО-МО, смены границ.

Выход: outputs/inventory/inventory.json (из него берутся числа об охвате данных в отчете и на лендинге).
Запуск: python scripts/inventory.py  (после scripts/download_data.py)
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import links  # noqa: E402
RAW = ROOT / "data" / "raw"
HACK = RAW / "hackathon" / "extracted" / "hackathonlicence"
OUT = ROOT / "outputs" / "inventory"


def q(s: pd.Series) -> dict:
    """Квантили для быстрого взгляда на распределение."""
    d = s.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_dict()
    return {k: (round(float(v), 3) if pd.notna(v) else None) for k, v in d.items()}


def consumption(res: dict) -> pd.DataFrame:
    df = pd.read_parquet(HACK / "consumption.parquet")
    r = {"rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()}}
    r["dates"] = sorted(df["date"].unique().tolist())
    r["n_months"] = len(r["dates"])
    r["categories"] = df["category"].value_counts().to_dict()
    ids = df["territory_id"].unique()
    r["n_territories"] = int(len(ids))
    r["duplicates_key"] = int(df.duplicated(["date", "territory_id", "category"]).sum())
    r["value_le_0"] = int((df["value"] <= 0).sum())
    r["value_na"] = int(df["value"].isna().sum())
    # Полнота панели: сколько МО имеют все месяцы в каждой категории
    cnt = df.groupby(["territory_id", "category"])["date"].nunique().unstack()
    r["territories_full_24m_by_cat"] = {c: int((cnt[c] == r["n_months"]).sum()) for c in cnt.columns}
    r["territories_any_gap"] = int((cnt.fillna(0) < r["n_months"]).any(axis=1).sum())
    r["territories_missing_category"] = {c: int(cnt[c].isna().sum()) for c in cnt.columns}
    per_month = df.groupby("date")["territory_id"].nunique()
    r["territories_per_month"] = {"min": int(per_month.min()), "max": int(per_month.max())}
    # Распределение значений по категориям (руб. в месяц на жителя, по описанию это средние траты)
    r["value_by_category"] = {c: q(g["value"]) for c, g in df.groupby("category")}
    # Сумма пяти категорий против «Все категории»: доля «прочего»
    wide = df.pivot_table(index=["territory_id", "date"], columns="category", values="value")
    five = [c for c in wide.columns if c != "Все категории"]
    share = wide[five].sum(axis=1, min_count=len(five)) / wide["Все категории"]
    r["share_5cats_in_total"] = q(share.dropna())
    r["n_5cats_gt_total"] = int((share > 1).sum())
    # Скачки: отношение месяца к медиане МО по категории (грубый детектор выбросов)
    med = df.groupby(["territory_id", "category"])["value"].transform("median")
    ratio = df["value"] / med
    r["jump_ratio_gt3"] = int((ratio > 3).sum())
    r["jump_ratio_lt_0_33"] = int((ratio < 1 / 3).sum())
    top = df.assign(ratio=ratio).sort_values("ratio", ascending=False).head(10)
    r["top_jumps"] = top[["territory_id", "date", "category", "value", "ratio"]].round(2).to_dict("records")
    # Сезонность «Все категории»: медиана по МО отношения месяца к среднему за год
    tot = df[df["category"] == "Все категории"].copy()
    tot["year"] = tot["date"].str[:4]
    tot["rel"] = tot["value"] / tot.groupby(["territory_id", "year"])["value"].transform("mean")
    r["seasonality_median_rel"] = tot.groupby("date")["rel"].median().round(3).to_dict()
    # Рост 2024/2023 «Все категории» и «Маркетплейсы»
    for c in ["Все категории", "Маркетплейсы"]:
        g = df[df["category"] == c].assign(year=lambda x: x["date"].str[:4])
        y = g.groupby(["territory_id", "year"])["value"].mean().unstack()
        r[f"growth_2024_2023_{c}"] = q((y["2024"] / y["2023"]).dropna())
    res["consumption"] = r
    return df


def api_consumption(res: dict, cons: pd.DataFrame, borders: pd.DataFrame) -> None:
    f = RAW / "sberindex_api" / "potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij.parquet"
    a = pd.read_parquet(f)
    r = {"rows": len(a), "columns": list(a.columns), "periods": [a["period"].min(), a["period"].max()],
         "n_periods": int(a["period"].nunique()), "n_mo_names": int(a["mo"].nunique()),
         "categories": sorted(a["category_15"].unique().tolist()), "freq": a["freq"].unique().tolist(),
         "unit": a["unit_measure"].unique().tolist()}
    # Совпадает ли с архивом хакатона: сравниваем отсортированные значения по месяцу и категории
    a2 = a.assign(date=a["period"].str[:7]).rename(columns={"category_15": "category"})
    s1 = cons.groupby(["date", "category"])["value"].apply(lambda x: tuple(sorted(x)))
    s2 = a2.groupby(["date", "category"])["value"].apply(lambda x: tuple(sorted(int(v) for v in x)))
    common = s1.index.intersection(s2.index)
    r["same_values_as_hackathon_share"] = round(float(np.mean([s1[i] == s2[i] for i in common])), 4) if len(common) else None
    res["api_consumption"] = r


def borders(res: dict) -> pd.DataFrame:
    b = pd.read_excel(RAW / "borders" / "extracted" / "t_dict_municipal_districts.xlsx")
    r = {"rows": len(b), "columns": list(b.columns), "n_territory_id": int(b["territory_id"].nunique()),
         "types": b["municipal_district_type"].value_counts().to_dict(),
         "year_from": b["year_from"].value_counts().sort_index().to_dict(),
         "year_to": b["year_to"].value_counts().sort_index().to_dict(),
         "n_regions": int(b["region_code"].nunique())}
    active = b[b["year_to"] == 9999]
    r["active_rows"] = len(active)
    r["active_unique_territory_id"] = int(active["territory_id"].nunique())
    changed = b[b["change_id_from"].notna() | b["change_id_to"].notna()]
    r["rows_with_change"] = len(changed)
    r["changes_since_2022"] = int(((b["year_from"] >= 2022) & (b["year_from"] < 9999)).sum())
    r["ended_2023_2025"] = int(b["year_to"].between(2023, 2025).sum())
    r["shape_values"] = b["shape"].value_counts().to_dict()
    r["example_changes"] = changed[["territory_id", "municipal_district_name", "year_from", "year_to",
                                    "change_id_from", "change_id_to"]].head(8).astype(str).to_dict("records")
    res["borders"] = r
    return b


def gpkg(res: dict) -> None:
    import pyogrio
    p = RAW / "borders" / "extracted" / "t_dict_municipal_districts_poly.gpkg"
    layers = pyogrio.list_layers(p)
    info = []
    for name, geom in layers:
        meta = pyogrio.read_info(p, layer=name)
        info.append({"layer": name, "geometry": geom, "features": int(meta["features"]),
                     "crs": str(meta["crs"]), "fields": list(meta["fields"])})
    res["gpkg"] = info


def contiguity(res: dict, cons_ids: set) -> None:
    """Соседство по общей границе (полигоны действующих МО): еще одна «реальная» связь."""
    import geopandas as gpd
    g = gpd.read_file(RAW / "borders" / "extracted" / "t_dict_municipal_districts_poly.gpkg")
    g = g[g["year_to"] == 9999][["territory_id", "geometry"]].reset_index(drop=True)
    g["territory_id"] = g["territory_id"].astype(int)
    pairs = links.contiguity_pairs(g).rename(columns={"a": "territory_id_left", "b": "territory_id_right"})
    nodes = set(pairs["territory_id_left"]) | set(pairs["territory_id_right"])
    deg = pd.concat([pairs["territory_id_left"], pairs["territory_id_right"]]).value_counts()
    res["contiguity"] = {"polygons_active": len(g), "pairs": len(pairs), "nodes_with_neighbors": len(nodes),
                         "isolated_polygons": int(len(set(g["territory_id"]) - nodes)),
                         "degree": q(deg), "consumption_ids_with_neighbors": len(cons_ids & nodes)}


def market_access(res: dict, cons_ids: set) -> None:
    m = pd.read_parquet(HACK / "market_access.parquet")
    ids = set(m["territory_id"])
    res["market_access"] = {"rows": len(m), "n_ids": len(ids), "value": q(m["market_access"]),
                            "na": int(m["market_access"].isna().sum()),
                            "consumption_ids_missing_here": len(cons_ids - ids),
                            "ids_not_in_consumption": len(ids - cons_ids)}


def pairs_summary(df: pd.DataFrame, ids_ref: set, label: str) -> dict:
    x, y = df["territory_id_x"].to_numpy(), df["territory_id_y"].to_numpy()
    nodes = set(np.unique(np.concatenate([x, y])))
    a, b = np.minimum(x, y), np.maximum(x, y)
    key = pd.Series(a.astype(np.int64) * 100000 + b)
    n = len(nodes)
    r = {"rows": len(df), "nodes": n, "self_loops": int((x == y).sum()),
         "unique_undirected_pairs": int(key.nunique()),
         "complete_graph_pairs": n * (n - 1) // 2,
         "duplicate_undirected_rows": int(key.duplicated().sum()),
         "distance": q(df["distance"]),
         "distance_le_0": int((df["distance"] <= 0).sum()),
         "consumption_ids_covered": len(ids_ref & nodes),
         "consumption_ids_missing": len(ids_ref - nodes)}
    r["density_vs_complete"] = round(r["unique_undirected_pairs"] / max(r["complete_graph_pairs"], 1), 4)
    # Противоречия: одна пара в обе стороны с разным расстоянием
    d = pd.DataFrame({"k": key, "d": df["distance"].to_numpy()})
    r["pairs_conflicting_distance"] = int((d.groupby("k")["d"].nunique() > 1).sum())
    # Неравенство треугольника не проверяем на всем графе (дорого); степень узлов:
    deg = pd.Series(np.concatenate([x, y])).value_counts()
    r["degree"] = q(deg)
    r["label"] = label
    return r


def connections(res: dict, cons_ids: set) -> None:
    c = pd.read_parquet(HACK / "connection.parquet")
    res["connection"] = {"types": c["type"].value_counts().to_dict(),
                         "by_type": {t: pairs_summary(g, cons_ids, t) for t, g in c.groupby("type")}}
    rw = pd.read_parquet(RAW / "railway" / "extracted" / "t_pairs_distance_railway.parquet")
    r = pairs_summary(rw, cons_ids, "railway_dataset")
    # Совпадает ли отдельный ж/д набор с railway из архива хакатона
    hr = c[c["type"] == "railway"]
    k1 = set(zip(np.minimum(hr.territory_id_x, hr.territory_id_y), np.maximum(hr.territory_id_x, hr.territory_id_y)))
    k2 = set(zip(np.minimum(rw.territory_id_x, rw.territory_id_y), np.maximum(rw.territory_id_x, rw.territory_id_y)))
    r["pairs_in_both"] = len(k1 & k2)
    r["pairs_only_here"] = len(k2 - k1)
    r["pairs_only_in_hackathon"] = len(k1 - k2)
    res["railway_dataset"] = r
    # Насколько дорожное расстояние больше прямого (извилистость), по центрам из справочника
    b = pd.read_excel(RAW / "borders" / "extracted" / "t_dict_municipal_districts.xlsx")
    b = b[b["year_to"] == 9999].drop_duplicates("territory_id").set_index("territory_id")
    h = c[c["type"] == "highway"].sample(200000, random_state=0)
    ok = h["territory_id_x"].isin(b.index) & h["territory_id_y"].isin(b.index)
    h = h[ok]
    la1, lo1 = np.radians(b.loc[h.territory_id_x, "municipal_district_center_lat"].to_numpy()), np.radians(b.loc[h.territory_id_x, "municipal_district_center_lon"].to_numpy())
    la2, lo2 = np.radians(b.loc[h.territory_id_y, "municipal_district_center_lat"].to_numpy()), np.radians(b.loc[h.territory_id_y, "municipal_district_center_lon"].to_numpy())
    gc = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    ratio = h["distance"].to_numpy() / np.maximum(gc, 1)
    res["connection"]["highway_detour_ratio_sample"] = q(pd.Series(ratio[gc > 20]))


def mobility(res: dict, borders_df: pd.DataFrame) -> None:
    m = pd.read_parquet(RAW / "sberindex_api" / "indeks-mobilnosti.parquet")
    names = set(borders_df.loc[borders_df["year_to"] == 9999, "municipal_district_name"])
    r = {"rows": len(m), "columns": list(m.columns), "periods": m["period"].value_counts().to_dict(),
         "n_areas": int(m["ref_area"].nunique()), "freq": m["freq"].unique().tolist(),
         "unit": m["unit_measure"].unique().tolist(), "value": q(m["value"]),
         "areas_matched_by_exact_name": int(m["ref_area"].drop_duplicates().isin(names).sum())}
    res["mobility"] = r


def rosstat(res: dict, borders_df: pd.DataFrame, cons_ids: set) -> None:
    d = RAW / "rosstat"
    r = {}
    if (d / "indicators.csv").exists():
        ind = pd.read_csv(d / "indicators.csv")
        r["n_indicators"] = len(ind)
        r["sections"] = ind.groupby("section")["indicator_section"].first().to_dict()
        r["indicators_per_section"] = ind["section"].value_counts().sort_index().to_dict()
        pat = re.compile(r"зарплат|заработн|численност|занят|ОКВЭД|отгруж|оборот|бюджет|доход|инвест|ввод|жиль|организац", re.I)
        r["relevant_indicators"] = ind[ind["indicator_name"].str.contains(pat, na=False)][["section", "indicator_code", "indicator_name", "unit"]].to_dict("records")
    files = sorted((d / "indicators").glob("*.csv")) if (d / "indicators").exists() else []
    act = borders_df[borders_df["year_to"] == 9999].copy()
    act["oktmo8"] = act["oktmo"].str.replace("-", "").str[:8]
    ok2tid = dict(zip(act["oktmo8"], act["territory_id"]))
    per = {}
    for f in files:
        x = pd.read_csv(f, sep=";", dtype=str)
        code = f.name.split("_")[1]
        e = per.setdefault(code, {"name": x["indicator_name"].iloc[0], "unit": x["indicator_unit"].iloc[0],
                                  "files": [], "years": {}, "columns": list(x.columns)})
        e["files"].append(f.name)
        for y, g in x.groupby("year"):
            ok = g["oktmo_stable"].fillna(g["oktmo"]).str[:8] if "oktmo_stable" in g else g["oktmo"].str[:8]
            tids = set(ok.map(ok2tid).dropna().astype(int))
            e["years"][y] = {"rows": len(g), "unique_oktmo": int(ok.nunique()),
                             "matched_territory_ids": len(tids), "consumption_ids_covered": len(tids & cons_ids)}
        extra = [c for c in x.columns if c not in ("indicator_section_code", "indicator_section", "indicator_code",
                 "indicator_name", "region_id", "region_name", "mun_level", "mun_district", "municipality", "oktmo",
                 "mun_type", "mun_type_oktmo", "oktmo_stable", "oktmo_history", "oktmo_year_from", "oktmo_year_to",
                 "year", "indicator_value", "indicator_unit", "indicator_period", "comment")]
        e["breakdown_columns"] = {c: x[c].dropna().unique()[:25].tolist() for c in extra}
    r["extracted"] = per
    res["rosstat"] = r


def extras(res: dict, cons: pd.DataFrame, b: pd.DataFrame) -> None:
    """Пропуски панели, пары «город + окружающий район», неоднозначность имен для мобильности."""
    act = b[b["year_to"] == 9999]
    ids = set(cons["territory_id"].astype(int))
    r = {}
    ended = b[b["territory_id"].isin(ids - set(act["territory_id"]))]
    r["consumption_ids_not_active"] = ended[["territory_id", "municipal_district_name", "year_to", "change_id_to"]].astype(str).to_dict("records")
    n = cons.groupby("territory_id")["date"].nunique()
    gap = n[n < 24]
    r["gap_months_hist"] = gap.value_counts().sort_index().to_dict()
    g = cons[cons["territory_id"].isin(gap.index)]
    first, last = g.groupby("territory_id")["date"].min(), g.groupby("territory_id")["date"].max()
    r["gap_start_after_2023_01"] = int((first > "2023-01").sum())
    r["gap_end_before_2024_12"] = int((last < "2024-12").sum())
    only = g.groupby("territory_id")["date"].agg(lambda s: "+".join(sorted(s.str[:4].unique())))
    r["gap_years_pattern"] = only.value_counts().to_dict()
    r["gap_regions_top"] = act[act["territory_id"].isin(gap.index)]["region_name"].value_counts().head(10).to_dict()
    full = n[n == 24].index
    r["full_panel_regions"] = int(act[act["territory_id"].isin(full)]["region_name"].nunique())
    r["regions_total_active"] = int(act["region_name"].nunique())
    r["regions_absent_in_full_panel"] = sorted(set(act["region_name"]) - set(act[act["territory_id"].isin(full)]["region_name"]))
    r["full_panel_types"] = act[act["territory_id"].isin(full)]["municipal_district_type"].value_counts().to_dict()
    # «Бублики»: shape 2/3 — район, центр которого лежит в соседнем городе
    rp = links.ring_pairs(act)
    pairs = {tuple(sorted((int(a), int(c)))) for a, c in zip(rp["ring"], rp["city"])}
    c = pd.read_parquet(HACK / "connection.parquet")
    z = c[(c["type"] == "highway") & (c["distance"] <= 0)]
    zp = {tuple(sorted((int(a), int(bb)))) for a, bb in zip(z["territory_id_x"], z["territory_id_y"])}
    r["ring_pairs"] = len(pairs)
    r["ring_pairs_both_in_full_panel"] = sum(1 for a, bb in pairs if a in full and bb in full)
    r["zero_highway_pairs"] = len(zp)
    r["zero_highway_pairs_that_are_ring"] = len(zp & pairs)
    # Мобильность: сопоставление по имени, одинаковые имена в разных регионах
    m = pd.read_parquet(RAW / "sberindex_api" / "indeks-mobilnosti.parquet")
    vc = act["municipal_district_name"].value_counts()
    amb = set(vc[vc > 1].index)
    names = m["ref_area"].drop_duplicates()
    ok = act[act["municipal_district_name"].isin(set(names) - amb)]
    r["mobility_names_ambiguous"] = int(names.isin(amb).sum())
    r["mobility_matched_unique"] = len(ok)
    r["mobility_matched_in_consumption"] = int(ok["territory_id"].isin(ids).sum())
    r["mobility_matched_in_full_panel"] = int(ok["territory_id"].isin(full).sum())
    r["mobility_types"] = ok["municipal_district_type"].value_counts().to_dict()
    r["mobility_regions_top"] = ok["region_name"].value_counts().head(8).to_dict()
    res["extras"] = r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    cons = consumption(res)
    ids = set(cons["territory_id"].astype(int))
    b = borders(res)
    res["consumption"]["ids_not_in_borders"] = len(ids - set(b["territory_id"]))
    res["consumption"]["ids_in_active_borders"] = len(ids & set(b.loc[b["year_to"] == 9999, "territory_id"]))
    api_consumption(res, cons, b)
    gpkg(res)
    contiguity(res, ids)
    market_access(res, ids)
    connections(res, ids)
    mobility(res, b)
    rosstat(res, b, ids)
    extras(res, cons, b)
    (OUT / "inventory.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: (list(v.keys()) if isinstance(v, dict) else len(v)) for k, v in res.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
