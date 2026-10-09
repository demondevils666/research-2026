"""Этап 4: сравнение методов на нескольких seed со статистическими тестами (в стиле работ Шалилеха).

Гиперпараметры фиксированы (как в основном сравнении, CANUS — лучший вариант из canus_tuning.json), меняется только seed.
Для каждого метода и seed — шесть ICVI конкурса (SW, CH, S_Dbw по признакам; AVI, AVU, Q на физической сети).
Блок = (seed, индекс): внутри блока методы ранжируются (1 — лучший с учетом направления индекса).
  тест Фридмана по блокам; критическая разность Немени (Demšar 2006) для диаграммы;
  попарно «KEFRiN, физическая сеть» против каждого метода: знаковый ранговый тест Уилкоксона по разностям рангов
  в блоках с поправкой Холма, d Коэна для парных разностей рангов;
  второй рейтинг — пороговое агрегирование (Алескеров): по каждому индексу метод получает оценку 1/2/3 (верхняя,
  средняя, нижняя треть методов по среднему за seed значению); методы упорядочиваются лексикографически по числу
  оценок «3», затем «2» (некомпенсаторно: провал по одному индексу не покрывается остальными).
Псевдоповторы: блоки «seed × индекс» не независимы (у детерминированных методов значения по seed одинаковы), поэтому
p-значение по 60 блокам завышено. Основной тест — Фридман по 6 блокам (индексам) на средних за seed значениях;
60-блочный вариант остается как дополнительный.
Выход: outputs/stage2/methods_seeds.json
Запуск: python scripts/methods_seeds.py [--post]  (--post: пересчитать только тесты по средним из готового json)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, icvi, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402

CONTEST = ["SW", "CH", "S_Dbw", "AVI", "AVU", "Q"]


def seed_mean_tests(mean_indices: dict, sd_indices: dict, alpha: float) -> dict:
    """Фридман и Немени по 6 блокам (индексам) на средних за seed значениях; какие методы детерминированы по seed."""
    names = list(mean_indices)
    M = pd.DataFrame(mean_indices).T.loc[names]
    R = np.array([M[kx].rank(ascending=not icvi.HIGHER_BETTER[kx]).to_numpy() for kx in CONTEST])
    fr = friedmanchisquare(*[R[:, j] for j in range(R.shape[1])])
    k, N = R.shape[1], R.shape[0]
    cd = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)) * np.sqrt(k * (k + 1) / (6 * N))
    return {"n_blocks": int(N), "mean_rank": dict(zip(names, R.mean(0).round(3).tolist())),
            "chi2": round(float(fr.statistic), 2), "p_value": float(fr.pvalue), "nemenyi_cd": round(float(cd), 3),
            "deterministic_methods": [m for m in names if all(v == 0 for v in sd_indices[m].values())]}


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * v))
        out[k], prev = adj, adj
    return out


def main() -> None:
    cfg = load()
    mc, ms = cfg["methods"], cfg["method_seeds"]
    proc, out = path("processed"), path("outputs") / "stage2"
    if "--post" in sys.argv:
        res = json.loads((out / "methods_seeds.json").read_text())
        res["friedman_seed_means"] = seed_mean_tests(res["mean_indices"], res["sd_indices"], ms["alpha"])
        (out / "methods_seeds.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res["friedman_seed_means"], ensure_ascii=False, indent=1))
        return
    K = json.loads((out / "select_k.json").read_text())["chosen_K"]
    alpha = json.loads((out / "kefrin_sweep.json").read_text())["chosen_alpha"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    e_phys, e_sim = pd.read_parquet(proc / "net_physical.parquet"), pd.read_parquet(proc / "net_cosine.parquet")
    Wp, Ws = networks.to_dense(e_phys, ids), networks.to_dense(e_sim, ids)
    Wg = networks.to_dense(networks.gaussian_knn(Y, ids, cfg["network"]["knn_k"]), ids)
    Zp, Zs = methods.kefrin_Z(Y, methods.network_matrix(Wp), alpha), methods.kefrin_Z(Y, methods.network_matrix(Ws), alpha)
    A = methods.diffusion(Wp, cfg["network"]["diffusion_steps"])
    tuned = out / "canus_tuning.json"
    cp = dict(json.loads(tuned.read_text())["chosen"]["params"]) if tuned.exists() else {"epochs": mc["canus_epochs"]}
    ep = cp.pop("epochs", mc["canus_epochs"])
    fits = {
        "kefrin_phys": lambda s: methods.kefrin(Y, None, K, alpha, s, mc["n_init"], Z=Zp)[0],
        "kefrin_sim": lambda s: methods.kefrin(Y, None, K, alpha, s, mc["n_init"], Z=Zs)[0],
        "kmeans": lambda s: methods.kmeans(Y, K, s, mc["n_init"]),
        "gmm": lambda s: GaussianMixture(K, covariance_type="full", n_init=3, random_state=s).fit_predict(Y),
        "spectral_sim": lambda s: methods.spectral(Wg, K, s),
        "leiden_phys": lambda s: methods.leiden_k(e_phys, ids, K, s)[0],
        "leiden_sim": lambda s: methods.leiden_k(e_sim, ids, K, s)[0],
        "canus_phys": lambda s: methods.canus(Y, A, K, s, ep, **cp),
    }
    rows = []
    for seed in ms["seeds"]:
        for name, f in fits.items():
            t = time.time()
            lab = np.asarray(f(seed))
            ind = icvi.all_indices(Y, Wp, lab)
            rows.append({"method": name, "seed": seed, "n_clusters": int(len(set(lab))), "time_s": round(time.time() - t, 1),
                         **{k: float(ind[k]) for k in CONTEST}})
            print(seed, name, {k: round(rows[-1][k], 3) for k in CONTEST}, rows[-1]["time_s"], "с", flush=True)
    df = pd.DataFrame(rows)
    names = list(fits)
    # Ранги внутри блоков (seed, индекс)
    blocks = []
    for seed, g in df.groupby("seed"):
        g = g.set_index("method").loc[names]
        for k in CONTEST:
            blocks.append(g[k].rank(ascending=not icvi.HIGHER_BETTER[k]).to_numpy())
    R = np.array(blocks)                       # блоки × методы
    mean_rank = dict(zip(names, R.mean(0).round(3).tolist()))
    fr = friedmanchisquare(*[R[:, j] for j in range(R.shape[1])])
    k, N = R.shape[1], R.shape[0]
    q = float(studentized_range.ppf(1 - ms["alpha"], k, np.inf) / np.sqrt(2))
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    ref = names.index(ms["reference"])
    pw, d = {}, {}
    for j, nm in enumerate(names):
        if j == ref:
            continue
        diff = R[:, j] - R[:, ref]               # > 0: эталон лучше (меньший ранг)
        pw[nm] = float(wilcoxon(diff, zero_method="zsplit").pvalue) if np.any(diff != 0) else 1.0
        d[nm] = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("inf")
    ph = holm(pw)
    # Пороговое агрегирование по средним за seed значениям
    mean_vals = df.groupby("method")[CONTEST].mean().loc[names]
    grades = pd.DataFrame(index=names, columns=CONTEST, dtype=int)
    for kx in CONTEST:
        r = mean_vals[kx].rank(ascending=not icvi.HIGHER_BETTER[kx], method="min")
        grades[kx] = np.where(r <= k / 3, 1, np.where(r <= 2 * k / 3, 2, 3))
    ta = sorted(names, key=lambda m: (int((grades.loc[m] == 3).sum()), int((grades.loc[m] == 2).sum())))
    res = {"definition": __doc__.split("Выход")[0].strip(), "K": K, "alpha": alpha, "seeds": ms["seeds"],
           "n_blocks": int(N), "mean_rank": mean_rank,
           "friedman": {"chi2": round(float(fr.statistic), 2), "p_value": float(fr.pvalue)},
           "nemenyi_cd": round(float(cd), 3), "cd_alpha": ms["alpha"],
           "reference": ms["reference"],
           "wilcoxon_vs_reference_holm": {m: float(f"{v:.3g}") for m, v in ph.items()},
           "cohen_d_rank_diff_vs_reference": {m: round(v, 2) for m, v in d.items()},
           "threshold_aggregation_order": ta,
           "threshold_grades": {m: {kx: int(grades.loc[m, kx]) for kx in CONTEST} for m in names},
           "mean_indices": {m: {kx: round(float(mean_vals.loc[m, kx]), 4) for kx in CONTEST} for m in names},
           "sd_indices": {m: {kx: round(float(v), 4) for kx, v in df[df["method"] == m][CONTEST].std().items()} for m in names},
           "n_clusters_range": {m: [int(df[df["method"] == m]["n_clusters"].min()), int(df[df["method"] == m]["n_clusters"].max())]
                                for m in names}}
    res["friedman_seed_means"] = seed_mean_tests(res["mean_indices"], res["sd_indices"], ms["alpha"])
    (out / "methods_seeds.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("definition", "threshold_grades", "mean_indices", "sd_indices")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
