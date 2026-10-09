"""Тесты разбора переписи 2010 (src/mo/census.py) на маленьких таблицах в формате тома 7."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import census  # noqa: E402


def _block(employed, own_region, own_settlement, other_region, unknown, kind="Городское и сельское население"):
    return [(kind, [employed, employed - 10]), ("в том числе занятые:", []),
            ("на территории своего субъекта", [own_region, own_region - 5]),
            ("из них на территории своего населенного пункта", [own_settlement, 1.0]),
            ("на территории другого субъекта", [other_region, 0.0]),
            ("на территории других стран", [0.0, 0.0]),
            ("без указания территории нахождения работы", [unknown, 0.0])]


def test_parse_share_and_blocks():
    lines = ([("9. ЗАНЯТОЕ В ЭКОНОМИКЕ НАСЕЛЕНИЕ ПО ТЕРРИТОРИИ НАХОЖДЕНИЯ РАБОТЫ", []), ("Энская область", [])]
             + _block(1000, 900, 800, 50, 50)                       # итог по региону: юнита еще нет
             + [("г. Энск", [])] + _block(500, 480, 470, 10, 10)
             + [("Энский муниципальный", []), ("район", [])]          # название на двух строках
             + _block(200, 190, 100, 5, 5) + _block(80, 75, 60, 3, 2, kind="Городское население")
             + [("Новоэнский", []), ("муниципальный район", [])]      # первая строка без вида МО
             + _block(100, 90, 30, 0, 10))
    df = census.parse(lines).set_index("unit")
    assert list(df.index) == ["г. Энск", "Энский муниципальный район", "Новоэнский муниципальный район"]
    # доля = (свой субъект − свой НП) / (занятые − без указания); блок «городское и сельское», а не «городское»
    assert df.loc["Энский муниципальный район", "share_other_settlement"] == pytest.approx(90 / 195)
    assert df.loc["Новоэнский муниципальный район", "share_other_settlement"] == pytest.approx(60 / 90)
    assert df.loc["г. Энск", "employed"] == 500


def test_parse_pdf_split_labels():
    # в PDF метка бывает разбита: «на территории своего» / «субъекта 900»; поле заполняется один раз
    lines = [("Энский район", []), ("Городское и сельское", []), ("население", [100.0]),
             ("на территории своего", []), ("субъекта", [90.0]),
             ("из них на территории своего", []), ("населенного пункта", [40.0]),
             ("без указания территории", []), ("нахождения работы", [0.0]),
             ("на территории своего", []), ("субъекта", [1.0])]
    df = census.parse(lines)
    assert df.loc[0, "own_region"] == 90 and df.loc[0, "share_other_settlement"] == pytest.approx(0.5)


def test_norm_name_and_match():
    assert census.norm_name("Городской округ г. Кызыл – г. Кызыл") == "кызыл"
    assert census.norm_name("городской округ город Кызыл") == "кызыл"
    assert census.norm_name("Усольское муниципальное образование, муниципальный район") == "усольский"
    assert census.norm_name("муниципальный район имени Лазо") == census.norm_name("Район им. Лазо") == "лазо"
    assert census.norm_name("Муниципальный район Бай-Тайгинский кожуун") == "бай-тайгинский"
    assert census.norm_name("Городской округ город Железногорск (ЗАТО)") == "железногорск"
    df = pd.DataFrame({"unit": ["Городской округ город Ачинск", "г. Ачинск", "Городской округ \"Город Мамоново\"",
                                "г. Чадан"], "share_other_settlement": [0.1, 0.2, 0.3, 0.4]})
    directory = pd.Series({1: "городской округ город Ачинск", 2: "городской округ Мамоновский", 3: "Абанский район"})
    hit, unmatched, missing = census.match(df, directory, {"мамоново": "мамоновский"})
    assert hit.set_index("territory_id")["share_other_settlement"].to_dict() == {1: 0.1, 2: 0.3}  # первая строка МО
    assert unmatched == ["г. Чадан"] and missing == ["Абанский район"]
