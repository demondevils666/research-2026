"""Загрузчики исходных данных (после scripts/download_data.py). Только чтение, без обработки."""

import pandas as pd

from .config import path


def _raw():
    return path("raw")


def hack(name: str) -> pd.DataFrame:
    """Таблица из архива хакатона: consumption, market_access, connection."""
    return pd.read_parquet(_raw() / "hackathon" / "extracted" / "hackathonlicence" / f"{name}.parquet")


def consumption() -> pd.DataFrame:
    return hack("consumption")


def borders() -> pd.DataFrame:
    """Версионный справочник МО: все версии (year_from, year_to; 9999 = действует)."""
    return pd.read_excel(_raw() / "borders" / "extracted" / "t_dict_municipal_districts.xlsx")


def borders_latest(b: pd.DataFrame | None = None) -> pd.DataFrame:
    """По одной (последней) версии на territory_id, индекс — territory_id."""
    b = borders() if b is None else b
    return b.sort_values("year_to").drop_duplicates("territory_id", keep="last").set_index("territory_id")


def polygons():
    import geopandas as gpd
    g = gpd.read_file(_raw() / "borders" / "extracted" / "t_dict_municipal_districts_poly.gpkg")
    g["territory_id"] = g["territory_id"].astype(int)
    return g


def oktmo8(s: pd.Series) -> pd.Series:
    """ОКТМО справочника ('79-701-000-000') -> 8 знаков, как в БД ПМО ('79701000')."""
    return s.astype(str).str.replace("-", "", regex=False).str[:8]


def rosstat_indicator(code: str, year: int) -> pd.DataFrame:
    f = _raw() / "rosstat" / "indicators" / f"data_{code}_year{year}_112_v20250918.csv"
    return pd.read_csv(f, sep=";", dtype=str)
