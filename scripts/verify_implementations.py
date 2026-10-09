"""Сверка наших реализаций с кодом авторов и библиотекой жюри — числа для отчета (то же, что проверяют тесты).

1) KEFRiN: наша реализация (k-means в пространстве [√ρ·Y | √ξ·P]) против кода авторов (KEFRiNe) на всех МО
   при выбранном K, сырая сеть с модулярной стандартизацией, как у авторов: ARI разбиений, отношение критериев, время.
2) Сетевые ICVI (AVI, AVU, ANUI, Q, density modularity) против Pattern (AdjacencyClusteringMetrics) на 5 случайных
   графах: максимальное относительное расхождение.
3) S_Dbw: вариант пакета s-dbw против самого пакета; вариант по статье Halkidi предпочитает истинное разбиение.
Код авторов и Pattern скачивает scripts/fetch_external.py в third_party/ (в репозиторий не входят).
Выход: outputs/stage2/verification.json
"""

import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import features, icvi, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402


def kefrin_vs_authors(n: int | None = None, K: int | None = None, alpha: float = 0.3) -> dict:
    p = ROOT / "third_party" / "kefrin"
    sys.path.insert(0, str(p))
    logging.disable(logging.CRITICAL)
    from kefrin import KEFRiNe
    proc = path("processed")
    ids = sorted(pd.read_parquet(proc / "mo.parquet").index)[:n]
    n = len(ids)
    K = K or json.loads((path("outputs") / "stage2" / "select_k.json").read_text())["chosen_K"]
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    e = pd.read_parquet(proc / "net_physical.parquet")
    W = networks.to_dense(e[e["a"].isin(ids) & e["b"].isin(ids)], ids)
    t = time.time()
    ours, _ = methods.kefrin(Y, W, K, alpha, seed=load()["seed"], steps=0)
    t_ours = time.time() - t
    P = methods.modularity_standardize(W)
    np.random.seed(load()["seed"])  # код авторов берет случайность из глобального генератора numpy
    t = time.time()
    ref = np.asarray(KEFRiNe(Y, P, rho=(1 - alpha) / (Y ** 2).sum(), xi=alpha / (P ** 2).sum(), n_clusters=K,
                             preprocessing_y="none", preprocessing_p="none"))
    t_ref = time.time() - t
    Z = methods.kefrin_space(Y, W, alpha, steps=0)

    def crit(lab):
        return sum(((Z[lab == k] - Z[lab == k].mean(0)) ** 2).sum() for k in np.unique(lab))

    return {"n_mo": n, "K": K, "alpha": alpha, "ari_ours_vs_authors": round(float(adjusted_rand_score(ours, ref)), 3),
            "criterion_ratio_ours_to_authors": round(float(crit(ours) / crit(ref)), 4),
            "time_s_ours": round(t_ours, 2), "time_s_authors": round(t_ref, 2),
            "speedup": round(t_ref / t_ours, 1) if t_ours > 0 else None}


def icvi_vs_pattern(n_graphs: int = 5) -> dict:
    p = ROOT / "third_party" / "pattern" / "pattern" / "metrics" / "clustering_metrics.py"
    spec = importlib.util.spec_from_file_location("pattern_cm", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    worst = 0.0
    for seed in range(n_graphs):
        rng = np.random.default_rng(seed)
        A = (rng.random((60, 60)) < 0.15) * rng.random((60, 60))
        A = np.triu(A, 1)
        W, y = A + A.T, rng.integers(0, 4, 60)
        ref = m.AdjacencyClusteringMetrics().get_metric(W, y)
        ours = icvi.graph_indices(W, y)
        for a, b in [("AVI", "AVI"), ("AVU", "AVU"), ("ANUI", "ANUI"), ("Q", "modularity"), ("Qdens", "density_modularity")]:
            worst = max(worst, abs(ours[a] - ref[b]) / max(abs(ref[b]), 1e-12))
    return {"n_graphs": n_graphs, "indices": ["AVI", "AVU", "ANUI", "Q (modularity)", "density modularity"],
            "max_relative_difference": float(f"{worst:.3g}")}


def s_dbw_check() -> dict:
    import s_dbw
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.3, (50, 3)), rng.normal(3, 0.3, (50, 3)), rng.normal([0, 3, 0], 0.3, (50, 3))])
    y = np.repeat([0, 1, 2], 50)
    ref = s_dbw.S_Dbw(X, y, method="Halkidi", centr="mean", nearest_centr=False)
    ours_pkg = icvi.s_dbw(X, y, variant="package")
    return {"package_variant_relative_difference": float(f"{abs(ours_pkg - ref) / abs(ref):.3g}"),
            "paper_variant_true_partition": round(float(icvi.s_dbw(X, y)), 4),
            "paper_variant_shuffled": round(float(icvi.s_dbw(X, rng.permutation(y))), 4)}


def main() -> None:
    res = {"definition": __doc__.split("Выход")[0].strip()}
    for k, f in (("kefrin", kefrin_vs_authors), ("graph_icvi_vs_pattern", icvi_vs_pattern), ("s_dbw", s_dbw_check)):
        try:
            res[k] = f()
        except Exception as ex:  # нет third_party/ — фиксируем причину, остальное считаем
            res[k] = {"error": repr(ex)[:200]}
        print(k, res[k], flush=True)
    (path("outputs") / "stage2" / "verification.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
