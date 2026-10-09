import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, interpret  # noqa: E402
from mo.config import load  # noqa: E402


def synthetic_wide(n_mo: int = 4, months=("2024-01", "2024-02", "2024-03"), seed: int = 0) -> pd.DataFrame:
    cfg = load()["panel"]
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([range(1, n_mo + 1), list(months)], names=["territory_id", "date"])
    w = pd.DataFrame(rng.uniform(500, 5000, (len(idx), len(cfg["categories"]))), index=idx, columns=cfg["categories"])
    w[cfg["other_category"]] = rng.uniform(1000, 8000, len(idx))
    w[cfg["total_category"]] = w.sum(axis=1)
    return w


def test_profile_partial_equals_profile_on_full_months():
    w = synthetic_wide()
    full = features.profile(w, "2024")
    part = features.profile_partial(w, features.median_log_level(w), "2024")
    assert np.allclose(full.to_numpy(), part[full.columns].to_numpy())


def test_profile_partial_uses_reference_median():
    w = synthetic_wide()
    one = w.loc[[1]]
    # уровень одного МО считается относительно медианы всей панели, а не самого себя
    p = features.profile_partial(one, features.median_log_level(w), "2024")
    assert abs(p.loc[1, "rel_level"]) > 0


def test_nearest_type_and_distance():
    ref = pd.DataFrame({"x": [0.0, 0.1, -0.1, 10.0, 10.1, 9.9], "y": [0.0, 0.1, -0.1, 10.0, 9.9, 10.1]}, index=range(6))
    labels = pd.Series([0, 0, 0, 1, 1, 1], index=range(6))
    P = pd.DataFrame({"x": [0.2, 9.8, 5.0], "y": [0.0, 10.0, 5.0]}, index=["a", "b", "c"])
    r = interpret.nearest_type(P, ref, labels)
    assert r.loc["a", "label"] == 0 and r.loc["b", "label"] == 1
    # точка посередине далеко от обоих центров — кандидат в «нетипичный профиль»
    thr = interpret.nearest_type(ref, ref, labels)["distance"].quantile(0.95)
    assert r.loc["c", "distance"] > thr
