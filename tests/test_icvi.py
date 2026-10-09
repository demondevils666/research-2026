"""Проверки ICVI: известные ответы на кликах, сверка с networkx, с библиотекой жюри Pattern и с пакетом s-dbw."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import icvi  # noqa: E402


def two_cliques(n=5):
    W = np.zeros((2 * n, 2 * n))
    W[:n, :n] = 1
    W[n:, n:] = 1
    np.fill_diagonal(W, 0)
    return W, np.array([0] * n + [1] * n)


def random_graph(n=60, p=0.15, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p) * rng.random((n, n))
    A = np.triu(A, 1)
    return A + A.T, rng.integers(0, 4, n)


def test_cliques_known_values():
    W, y = two_cliques()
    g = icvi.graph_indices(W, y)
    assert g["AVI"] == pytest.approx(1.0)
    assert g["AVU"] == pytest.approx(0.0)
    assert g["Q"] == pytest.approx(0.5)
    assert g["TurboMQ"] == pytest.approx(2.0)


def test_turbomq_equals_k_avi_on_undirected():
    W, y = random_graph()
    g = icvi.graph_indices(W, y)
    assert g["TurboMQ"] == pytest.approx(4 * g["AVI"])


def test_modularity_matches_networkx():
    nx = pytest.importorskip("networkx")
    W, y = random_graph()
    G = nx.from_numpy_array(W)
    comms = [set(np.where(y == k)[0]) for k in np.unique(y)]
    assert icvi.graph_indices(W, y)["Q"] == pytest.approx(nx.community.modularity(G, comms, weight="weight"))


def test_matches_pattern_library():
    """Сверка с AdjacencyClusteringMetrics из Pattern (скачивается scripts/fetch_external.py)."""
    p = ROOT / "third_party" / "pattern"
    if not p.exists():
        pytest.skip("third_party/pattern нет (scripts/fetch_external.py)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("pattern_cm", p / "pattern" / "metrics" / "clustering_metrics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for seed in range(5):
        W, y = random_graph(seed=seed)
        ref = m.AdjacencyClusteringMetrics().get_metric(W, y)
        ours = icvi.graph_indices(W, y)
        for a, b in [("AVI", "AVI"), ("AVU", "AVU"), ("ANUI", "ANUI"), ("Q", "modularity"), ("Qdens", "density_modularity")]:
            assert ours[a] == pytest.approx(ref[b], rel=1e-9, abs=1e-12), (seed, a)


def test_s_dbw_matches_package_and_prefers_true_partition():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.3, (50, 3)), rng.normal(3, 0.3, (50, 3)), rng.normal([0, 3, 0], 0.3, (50, 3))])
    y = np.repeat([0, 1, 2], 50)
    ours = icvi.s_dbw(X, y)
    assert ours < icvi.s_dbw(X, rng.permutation(y))
    s_dbw = pytest.importorskip("s_dbw")
    ref = s_dbw.S_Dbw(X, y, method="Halkidi", centr="mean", nearest_centr=False)
    assert icvi.s_dbw(X, y, variant="package") == pytest.approx(ref, rel=1e-9)


def test_random_baseline_avi_near_1_over_k():
    W, _ = random_graph(n=200, p=0.05, seed=3)
    y = np.repeat(np.arange(4), 50)
    X = np.random.default_rng(0).normal(size=(200, 3))
    rb = icvi.random_baseline(X, W, y, n=20, seed=0)
    assert rb["AVI"]["mean"] == pytest.approx(0.25, abs=0.05)
