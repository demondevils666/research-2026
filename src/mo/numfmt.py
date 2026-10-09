"""Единое округление чисел для отчета, рисунков и лендинга.

Половина округляется вверх от кратчайшей десятичной записи числа (как Intl.NumberFormat в браузере):
1,355 -> 1,36 и в PDF, и на лендинге. Обычное f"{x:.2f}" округляет двоичное значение (1,35499…) и дает 1,35.
"""

from decimal import ROUND_HALF_UP, Decimal


def fixed(x: float, d: int) -> str:
    """Строка с d знаками после точки (точка, без разделителей тысяч)."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:f}"
    return "0" + s[2:] if s.startswith("-0") and set(s[2:].replace(".", "")) <= {"0"} else s
