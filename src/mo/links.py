"""Физические связи между МО: общая граница, пары «город + окружающий район», ближайшие по дороге.

Это не «похожесть» по тратам, а независимый от признаков источник: с кем МО связан географически.
"""

import numpy as np
import pandas as pd

from . import data


def contiguity_pairs(polys) -> pd.DataFrame:
    """Пары МО с общей границей (пересечение полигонов). Колонки a < b."""
    import geopandas as gpd
    g = polys[["territory_id", "geometry"]].reset_index(drop=True).copy()
    g["geometry"] = g.geometry.make_valid().buffer(0)
    p = gpd.sjoin(g, g, predicate="intersects")
    p = p[p["territory_id_left"] < p["territory_id_right"]]
    return pd.DataFrame({"a": p["territory_id_left"].to_numpy(), "b": p["territory_id_right"].to_numpy()})


def contiguity(ids: set | None = None) -> pd.DataFrame:
    """Соседство по последней версии полигона каждого МО (для МО, упраздненных в 2024, это версия 2023)."""
    g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last")
    if ids is not None:
        g = g[g["territory_id"].isin(ids)]
    return contiguity_pairs(g)


def ring_pairs(b_active: pd.DataFrame) -> pd.DataFrame:
    """Пары «окружающий район — город-центр» по shape_linked_oktmo.

    ВНИМАНИЕ: в данных коды shape обратны описанию в PDF-метаданных справочника. В данных shape=2 стоит
    у города (Томск, 115 из 116 — городские округа), а shape=3 — у района вокруг него (Томский район,
    71 из 75 — районы и муниципальные округа). Поэтому ориентируемся на данные: city — член пары с shape=2,
    ring — с shape=3.
    b_active: действующие версии справочника (колонки territory_id, oktmo, shape, shape_linked_oktmo).
    Возвращает колонки ring, city (territory_id), без дублей.
    """
    o2t = dict(zip(b_active["oktmo"], b_active["territory_id"]))
    shape = dict(zip(b_active["territory_id"], b_active["shape"]))
    rows = set()
    for _, r in b_active[b_active["shape"].isin([2, 3])].iterrows():
        for o in str(r["shape_linked_oktmo"]).replace(";", ",").split(","):
            other = o2t.get(o.strip())
            if other is None:
                continue
            a, c = int(r["territory_id"]), int(other)
            if {shape.get(a), shape.get(c)} != {2, 3}:
                continue
            city, ring = (a, c) if shape[a] == 2 else (c, a)
            rows.add((ring, city))
    return pd.DataFrame(sorted(rows), columns=["ring", "city"])


def road_knn(ids: set, k: int) -> pd.DataFrame:
    """k ближайших по дороге МО для каждого МО из ids (только внутри ids). Колонки a, b, km (a < b)."""
    c = data.hack("connection")
    c = c[(c["type"] == "highway") & c["territory_id_x"].isin(ids) & c["territory_id_y"].isin(ids)]
    both = pd.concat([
        pd.DataFrame({"src": c["territory_id_x"], "dst": c["territory_id_y"], "km": c["distance"]}),
        pd.DataFrame({"src": c["territory_id_y"], "dst": c["territory_id_x"], "km": c["distance"]}),
    ])
    near = both.sort_values(["src", "km"]).groupby("src").head(k)
    a, b = np.minimum(near["src"], near["dst"]), np.maximum(near["src"], near["dst"])
    out = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy(), "km": near["km"].to_numpy()})
    return out.drop_duplicates(["a", "b"]).reset_index(drop=True)
