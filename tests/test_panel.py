"""Проверки панели и признаков этапа 1. Нужен прогон scripts/build_panel.py (data/processed/)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import features, rosstat  # noqa: E402
from mo.config import path  # noqa: E402

PROC = path("processed")
pytestmark = pytest.mark.skipif(not (PROC / "panel_wide.parquet").exists(), reason="сначала scripts/build_panel.py")


@pytest.fixture(scope="module")
def w():
    return pd.read_parquet(PROC / "panel_wide.parquet")


@pytest.fixture(scope="module")
def mo():
    return pd.read_parquet(PROC / "mo.parquet")


def test_panel_shape(w):
    assert w.index.get_level_values("territory_id").nunique() == 2016
    assert w.index.get_level_values("date").nunique() == 24
    assert len(w) == 2016 * 24
    assert not w.isna().any().any()


def test_other_nonnegative_and_shares_sum_to_one(w):
    assert (w["Прочее"] >= 0).all()
    sh = features.shares(w)
    assert np.allclose(sh.sum(axis=1), 1.0)


def test_clr_rows_sum_to_zero(w):
    fm = features.monthly(w)
    clr = fm[[c for c in fm.columns if c.startswith("clr_")]]
    assert np.allclose(clr.sum(axis=1), 0.0, atol=1e-9)
    # относительный уровень: медиана по МО в каждом месяце равна нулю
    assert np.allclose(fm["rel_level"].groupby(level="date").median(), 0.0)


def test_rosstat_coverage(mo):
    assert mo["population"].notna().mean() >= 0.98
    assert (mo["population"] > 0).all()
    assert mo["wage"].notna().mean() >= 0.98
    blocks = [c for c in mo.columns if c.startswith("emp_") and c != "emp_covered"]
    assert (mo[blocks].sum(axis=1) <= 1.01).all()


def test_oktmo_map_prefers_version_active_in_year():
    # Сочи: ОКТМО 03726000 у territory_id 628 (2018–2021) и 2812 (с 2021)
    b = pd.DataFrame({"oktmo": ["03-726-000-000", "03-726-000-000"], "territory_id": [628, 2812],
                      "year_from": [2018, 2021], "year_to": [2021, 9999]})
    assert rosstat.oktmo_map(b, 2020)["03726000"] == 628
    assert rosstat.oktmo_map(b, 2023)["03726000"] == 2812


def test_ring_pairs_orientation(mo):
    rp = pd.read_parquet(PROC / "ring_pairs.parquet")
    assert len(rp) == 65
    assert not rp.duplicated().any()
    # «бублик» — район или округ (не центр), центр — город
    assert (mo.loc[rp["city"], "shape"] == 2).all() and (mo.loc[rp["ring"], "shape"] == 3).all()
    assert mo.loc[rp["city"], "type"].isin(["городской округ", "внутригородская территория города федерального значения"]).all()
    assert (mo.loc[rp["ring"], "type"] != "внутригородская территория города федерального значения").all()


def test_contiguity_edges_valid(mo):
    e = pd.read_parquet(PROC / "edges_contiguity.parquet")
    assert (e["a"] < e["b"]).all()
    assert e["a"].isin(mo.index).all() and e["b"].isin(mo.index).all()
    assert not e.duplicated().any()
