import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import numfmt  # noqa: E402


def test_half_up_from_shortest_repr():
    # как Intl.NumberFormat в браузере: половина вверх от десятичной записи, а не от двоичного значения
    assert numfmt.fixed(1.355, 2) == "1.36"
    assert numfmt.fixed(0.125, 2) == "0.13"
    assert numfmt.fixed(2.5, 0) == "3"
    assert numfmt.fixed(-2.5, 0) == "-3"
    assert numfmt.fixed(-0.243, 2) == "-0.24"


def test_negative_zero_is_plain_zero():
    assert numfmt.fixed(-0.0004, 3) == "0.000"
    assert numfmt.fixed(-0.00001, 2) == "0.00"
