"""KEFRiN: наша реализация (k-means в пространстве KEFRiN) против кода авторов на подвыборке реальных данных."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import features, methods, networks  # noqa: E402
from mo.config import path  # noqa: E402

PROC = path("processed")


def _data(n=500):
    if not (PROC / "net_physical.parquet").exists():
        pytest.skip("сначала scripts/build_panel.py и scripts/run_networks.py")
    mo = pd.read_parquet(PROC / "mo.parquet")
    ids = sorted(mo.index)[:n]
    Y = features.standardize(pd.read_parquet(PROC / "profile.parquet")).loc[ids].to_numpy()
    e = pd.read_parquet(PROC / "net_physical.parquet")
    e = e[e["a"].isin(ids) & e["b"].isin(ids)]
    return Y, networks.to_dense(e, ids)


def test_alpha_zero_is_kmeans_objective():
    Y, W = _data()
    Z = methods.kefrin_space(Y, W, 0.0)
    assert np.allclose(Z[:, Y.shape[1]:], 0)
    assert np.allclose(Z[:, :Y.shape[1]] / Z[0, 0] * Y[0, 0], Y)


def test_matches_authors_kefrin():
    p = ROOT / "third_party" / "kefrin"
    if not p.exists():
        pytest.skip("third_party/kefrin нет (scripts/fetch_external.py)")
    sys.path.insert(0, str(p))
    logging.disable(logging.CRITICAL)
    from kefrin import KEFRiNe
    Y, W = _data()
    alpha = 0.3
    ours, _ = methods.kefrin(Y, W, 4, alpha, seed=42, steps=0)  # сырая сеть, как в коде авторов
    P = methods.modularity_standardize(W)
    np.random.seed(42)  # код авторов берет случайность из глобального генератора numpy: без seed тест нестабилен
    ref = np.asarray(KEFRiNe(Y, P, rho=(1 - alpha) / (Y ** 2).sum(), xi=alpha / (P ** 2).sum(), n_clusters=4,
                             preprocessing_y="none", preprocessing_p="none"))
    Z = methods.kefrin_space(Y, W, alpha, steps=0)

    def crit(lab):
        return sum(((Z[lab == k] - Z[lab == k].mean(0)) ** 2).sum() for k in np.unique(lab))

    assert adjusted_rand_score(ours, ref) > 0.85
    assert crit(ours) <= crit(ref) * 1.01


def test_monthly_layer_limits_and_evolve_with_constant_dict():
    import pandas as pd
    from mo import dynamics, methods, networks
    rng = np.random.default_rng(3)
    ids = list(range(30))
    edges = pd.DataFrame({"a": np.arange(29), "b": np.arange(1, 30), "w": rng.uniform(0.2, 1, 29)})
    Ym = {f"2024-0{m}": rng.normal(size=(30, 4)) for m in range(1, 4)}
    layers, h = networks.monthly_layer(edges, ids, Ym, h=1e9)
    for L in layers.values():  # h → ∞: слой совпадает с физической сетью
        assert np.allclose(L["w"], edges["w"]) and np.allclose(L["sim"], 1.0)
    layers, h = networks.monthly_layer(edges, ids, Ym)
    assert h > 0 and all((L["w"] <= edges["w"] + 1e-12).all() for L in layers.values())
    P = methods.network_matrix(networks.to_dense(edges, ids))
    static = rng.integers(0, 3, 30)
    a = dynamics.evolve(Ym, P, static, 3, 0.5, 0.5, 42)
    b = dynamics.evolve(Ym, {d: P for d in Ym}, static, 3, 0.5, 0.5, 42)
    assert all((a["labels"][d] == b["labels"][d]).all() for d in Ym)


def test_moved_by_network_counts_border_mo():
    # две клики по 3 МО; МО 3 без сети попал не к своим соседям, с сетью — к ним
    ids = [1, 2, 3, 4, 5, 6]
    e = pd.DataFrame({"a": [1, 1, 2, 4, 4, 5, 3], "b": [2, 3, 3, 5, 6, 6, 4], "w": [1, 1, 1, 1, 1, 1, 0.1]})
    lab0 = np.array([0, 0, 1, 1, 1, 1])
    lab1 = np.array([0, 0, 0, 1, 1, 1])
    r = methods.moved_by_network(lab0, lab1, ids, e)
    assert r["n_moved"] == 1
    assert r["own_before"] == pytest.approx(0.1 / 2.1)
    assert r["own_after"] == pytest.approx(2 / 2.1)
