"""Интерпретация: правило Миркина, η², якоря названий типов."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import interpret  # noqa: E402
from mo.config import path  # noqa: E402


def test_mirkin_and_eta2_known_values():
    lab = pd.Series([0, 0, 1, 1])
    x = pd.DataFrame({"v": [1.0, 1.0, 3.0, 3.0]})
    m = interpret.mirkin(x, lab)
    assert m.loc[0, "v"] == pytest.approx(-50.0) and m.loc[1, "v"] == pytest.approx(50.0)
    assert interpret.eta2(x["v"], lab)["eta2"] == pytest.approx(1.0)
    assert interpret.eta2(pd.Series([1.0, 3.0, 1.0, 3.0]), lab)["eta2"] == pytest.approx(0.0)


def test_type_anchors_match_labels():
    f = path("outputs") / "stage2" / "labels_static.csv"
    if not f.exists():
        pytest.skip("нет итоговой типологии (scripts/run_compare.py)")
    lab = pd.read_csv(f, index_col="territory_id")["type"]
    t = interpret.type_names(lab)
    K = lab.nunique()                       # число типов — из итоговой типологии, названия — из configs/types.yaml
    assert sorted(t["code"]) == [f"T{i}" for i in range(1, K + 1)]
    assert t.index.nunique() == K


def test_type_names_raise_on_wrong_anchor():
    lab = pd.Series({1597: 0}, name="type")
    with pytest.raises(ValueError):
        interpret.type_names(lab)


def test_interval_rules_recover_planted_rule():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 400), "b": rng.uniform(0, 10, 400), "c": rng.uniform(0, 10, 400)})
    lab = pd.Series(np.where((X["a"] >= 5) & (X["b"] <= 3), 1, 0))
    qs = [i / 10 for i in range(1, 10)]
    res = interpret.interval_rules(X, lab, qs, {"a": 0.5, "b": 0.5, "c": 0.5}, max_terms=3, folds=3)
    r = res[1]
    feats = {(t["feature"], t["op"]) for t in r["rule"]}
    assert ("a", ">=") in feats and ("b", "<=") in feats
    assert r["f1"] > 0.9 and r["f1_cv_mean"] > 0.8
    assert r["precision"] > r["prevalence"]


def test_ordinal_patterns_separate_opposite_orders():
    rng = np.random.default_rng(2)
    a = rng.normal([1.0, 0.5, 0.0], 0.05, (30, 3))
    b = rng.normal([0.0, 0.5, 1.0], 0.05, (30, 3))
    dev = pd.DataFrame(np.vstack([a, b]), columns=["x", "y", "z"])
    lab = pd.Series(["A"] * 30 + ["B"] * 30)
    r = interpret.ordinal_patterns(dev, lab, eps=0.05)
    assert r["closest_is_own_share_all"] == 1.0
    assert r["type_pattern_hamming"]["A-B"] == r["n_pairs"]


def test_eta2_diff_ci_matches_eta2():
    rng = np.random.default_rng(0)
    g = pd.Series(rng.integers(0, 3, 300))
    x = pd.Series(g.to_numpy() * 2.0 + rng.normal(size=300))
    noise = pd.Series(rng.integers(0, 3, 300))
    assert interpret._eta2_codes(x.to_numpy(), g.to_numpy(), 3) == pytest.approx(interpret.eta2(x, g)["eta2"])
    lo, hi = interpret.eta2_diff_ci(x, g, noise, n=200, seed=1)
    assert lo > 0 and hi > lo
