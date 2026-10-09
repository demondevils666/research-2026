"""Этап 2, шаг 2: выбор числа типов K по бутстрап-устойчивости и кривым ICVI.

Для k-means (только признаки) и KEFRiN (признаки + физическая сеть, α из configs/base.yaml) при каждом K:
20 подвыборок по 80% МО -> ARI с разбиением полной выборки (на общих МО) и покластерный Жаккар (Hennig 2007).
Правило: наибольшее K, у которого медианный ARI >= stability_ari_min и минимальный средний Жаккар >= stability_jaccard_min
(для главного метода KEFRiN). Плюс ICVI на полной выборке для каждого K.
Выход: outputs/stage2/select_k.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, icvi, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402


def fit(method: str, Y, W, K, seed, alpha, n_init):
    if method == "kmeans":
        return methods.kmeans(Y, K, seed, n_init)
    return methods.kefrin(Y, W, K, alpha, seed, n_init)[0]


def hennig_jaccard(full: np.ndarray, sub_idx: np.ndarray, sub_lab: np.ndarray) -> np.ndarray:
    """Для каждого кластера полной выборки — максимальный Жаккар с кластерами подвыборки (на общих МО)."""
    f = full[sub_idx]
    out = []
    for k in np.unique(full):
        A = set(np.where(f == k)[0])
        best = max((len(A & set(np.where(sub_lab == j)[0])) / len(A | set(np.where(sub_lab == j)[0]))
                    for j in np.unique(sub_lab)), default=0.0)
        out.append(best)
    return np.array(out)


def subsamples(n: int, mc: dict, seed: int) -> list[np.ndarray]:
    """Подвыборки для проверки устойчивости — одни и те же в выборе K, развертке α и проверке K при итоговом α."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n, int(mc["bootstrap_frac"] * n), replace=False)) for _ in range(mc["bootstrap"])]


def stability(method, Y, W, K, alpha, subs, seed, n_init, full=None):
    """Устойчивость разбиения: медианный и 10%-й ARI полной подгонки с подгонками на подвыборках и покластерный
    Жаккар (Hennig 2007), усредненный по подвыборкам. Возвращает (полное разбиение, ARI медиана, ARI q10, Жаккар)."""
    if full is None:
        full = fit(method, Y, W, K, seed, alpha, n_init)
    aris, jac = [], []
    for s in subs:
        sub = fit(method, Y[s], W[np.ix_(s, s)], K, seed, alpha, max(3, n_init // 2))
        aris.append(adjusted_rand_score(full[s], sub))
        jac.append(hennig_jaccard(full, s, sub))
    return full, float(np.median(aris)), float(np.quantile(aris, 0.1)), np.mean(jac, axis=0)


def is_stable(ari_median: float, jaccard_min: float, mc: dict) -> bool:
    """Правило устойчивости — общее для выбора K и выбора α."""
    return ari_median >= mc["stability_ari_min"] and jaccard_min >= mc["stability_jaccard_min"]


def main() -> None:
    cfg = load()
    mc, seed = cfg["methods"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage2"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    W = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    subs = subsamples(len(ids), mc, seed)
    res = {"rule": __doc__.split("Выход")[0].strip(), "alpha": mc["kefrin_alpha"], "by_method": {}}
    t0 = time.time()
    for method in ("kmeans", "kefrin"):
        rows = []
        for K in mc["k_grid"]:
            full, ari_med, ari_q10, jac = stability(method, Y, W, K, mc["kefrin_alpha"], subs, seed, mc["n_init"])
            ind = icvi.all_indices(Y, W, full)
            rows.append({"K": K, "ari_median": round(ari_med, 3), "ari_q10": round(ari_q10, 3),
                         "jaccard_min": round(float(jac.min()), 3), "jaccard_by_cluster": [round(float(v), 3) for v in jac],
                         "sizes": np.bincount(full).tolist(), **{k: round(float(v), 4) for k, v in ind.items()}})
            print(method, K, rows[-1]["ari_median"], rows[-1]["jaccard_min"], f"{time.time() - t0:.0f}s", flush=True)
        res["by_method"][method] = rows
    ok = [r["K"] for r in res["by_method"]["kefrin"] if is_stable(r["ari_median"], r["jaccard_min"], mc)]
    res["chosen_K"] = max(ok) if ok else None
    res["stable_K_kefrin"] = ok
    res["stable_K_kmeans"] = [r["K"] for r in res["by_method"]["kmeans"] if is_stable(r["ari_median"], r["jaccard_min"], mc)]
    (out / "select_k.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for m in ("kmeans", "kefrin"):
        print(m)
        print(pd.DataFrame(res["by_method"][m])[["K", "ari_median", "ari_q10", "jaccard_min", "SW", "CH", "S_Dbw", "AVI", "AVU", "Q", "sizes"]].to_string(index=False))
    print("chosen K:", res["chosen_K"], "stable kefrin:", ok, "stable kmeans:", res["stable_K_kmeans"])


if __name__ == "__main__":
    main()
