import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo.names import short  # noqa: E402


def test_cities_lose_prefix():
    assert short("городской округ город Майкоп") == "Майкоп"
    assert short("городской округ город-курорт Железноводск") == "Железноводск"
    assert short("городской округ город-герой Новороссийск") == "Новороссийск"


def test_adjective_names_keep_kind():
    # без «г. о.» / «м. о.» не понять, город это или район: Пермский г. о. и Пермский м. о. — разные МО
    assert short("городской округ Талдомский") == "Талдомский г. о."
    assert short("муниципальный округ Гурьевский") == "Гурьевский м. о."
    assert short("муниципальный округ Завьяловский район") == "Завьяловский район"


def test_districts_and_federal_cities():
    assert short("Иркутский муниципальный район") == "Иркутский р-н"
    assert short("Благовещенский муниципальный округ") == "Благовещенский м. о."
    assert short("внутригородское муниципальное образование города федерального значения "
                 "муниципальный округ Арбат") == "Арбат"
