import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo.dynamics import monic  # noqa: E402


def lab(*blocks):
    """Метки из блоков (метка, число МО) подряд."""
    return np.concatenate([np.full(n, k) for k, n in blocks])


def test_survive_with_boundary_noise():
    prev = lab((0, 10), (1, 10))
    nxt = lab((0, 9), (1, 1), (1, 10))          # один МО типа 0 перешел в тип 1
    r = monic(prev, nxt, 0.5, 0.25)
    assert r["survive"] == {0: 0, 1: 1}
    assert r["overlap"] == {0: 0.9, 1: 1.0}
    assert not (r["split"] or r["absorb"] or r["disappear"] or r["emerge"])


def test_split_and_emerge():
    prev = lab((0, 10), (1, 10))
    nxt = lab((0, 10), (1, 4), (2, 4), (3, 2))  # тип 1 делится на 1 и 2 (по 40%), доли меньше порога выживания
    r = monic(prev, nxt, 0.5, 0.25)
    assert r["split"] == {1: [1, 2]}
    assert r["survive"] == {0: 0}
    assert r["emerge"] == [3]                   # 20% — меньше порога части расщепления


def test_absorb():
    prev = lab((0, 10), (1, 10), (2, 10))
    nxt = lab((0, 20), (2, 10))                 # типы 0 и 1 слились
    r = monic(prev, nxt, 0.5, 0.25)
    assert r["absorb"] == {0: [0, 1]}
    assert r["survive"] == {2: 2}


def test_disappear():
    prev = lab((0, 10), (1, 9))
    nxt = lab((0, 10), (0, 3), (2, 2), (3, 2), (4, 2))  # тип 1 рассыпался мелкими частями
    r = monic(prev, nxt, 0.5, 0.25)
    assert r["disappear"] == [1]
