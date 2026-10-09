"""Этап 2, шаг 4: сравнение методов при выбранном K (select_k.json) и α* (kefrin_sweep.json).

Методы: k-means и GMM (только признаки); спектральная (граф сходства по признакам); Leiden (только сеть) на физической
сети и на сети сходства; KEFRiN на физической сети (главный) и на сети сходства по признакам; CANUS (код автора, параметры — лучшие
из сетки scripts/canus_tuning.py)
на физической сети.
Для каждого: 6 ICVI конкурса (SW, CH, S_Dbw; AVI, AVU, MQ = Q на физической сети) + ANUI, BasicMQ, TurboMQ и индексы на сети
сходства; z-оценка против случайной перестановки меток; связность агломераций; ARI с регионами; бутстрап-устойчивость.
Выход: outputs/stage2/compare.json, outputs/stage2/labels_static.csv (итоговая статическая типология и все методы)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, features, icvi, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402

CONTEST = ["SW", "CH", "S_Dbw", "AVI", "AVU", "Q"]


def bootstrap_ari(fit, n: int, B: int, frac: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    full = fit(np.arange(n))
    vals = []
    for _ in range(B):
        s = np.sort(rng.choice(n, int(frac * n), replace=False))
        vals.append(adjusted_rand_score(full[s], fit(s)))
    return float(np.median(vals))


def main() -> None:
    cfg = load()
    seed, mc = cfg["seed"], cfg["methods"]
    proc, out = path("processed"), path("outputs") / "stage2"
    K = json.loads((out / "select_k.json").read_text())["chosen_K"]
    alpha = json.loads((out / "kefrin_sweep.json").read_text())["chosen_alpha"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    n = len(ids)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    e_phys = pd.read_parquet(proc / "net_physical.parquet")
    e_sim = pd.read_parquet(proc / "net_cosine.parquet")
    Wp, Ws = networks.to_dense(e_phys, ids), networks.to_dense(e_sim, ids)
    Wg = networks.to_dense(networks.gaussian_knn(Y, ids, cfg["network"]["knn_k"]), ids)
    Pp, Ps = methods.network_matrix(Wp), methods.network_matrix(Ws)
    u = agglo.units(mo, pd.read_parquet(proc / "ring_pairs.parquet"), pd.read_parquet(proc / "edges_contiguity.parquet"))
    region = mo.loc[ids, "region_code"].to_numpy()

    def sub_kefrin(P_full, W_full):
        def f(s):
            if len(s) == n:
                return methods.kefrin(Y, W_full, K, alpha, seed, mc["n_init"], Z=methods.kefrin_Z(Y, P_full, alpha))[0]
            return methods.kefrin(Y[s], W_full[np.ix_(s, s)], K, alpha, seed, max(3, mc["n_init"] // 2))[0]
        return f

    fits = {
        "kmeans": (lambda s: methods.kmeans(Y[s], K, seed, mc["n_init"]), "признаки"),
        "gmm": (lambda s: GaussianMixture(K, covariance_type="full", n_init=3, random_state=seed).fit_predict(Y[s]), "признаки"),
        "spectral_sim": (lambda s: methods.spectral(Wg[np.ix_(s, s)], K, seed), "граф сходства по признакам"),
        "kefrin_phys": (sub_kefrin(Pp, Wp), f"признаки + физическая сеть (α = {alpha})"),
        "kefrin_sim": (sub_kefrin(Ps, Ws), f"признаки + сеть сходства (α = {alpha})"),
    }
    labels, res = {}, {"K": K, "alpha": alpha, "methods": {}}
    for name, (f, desc) in fits.items():
        t = time.time()
        labels[name] = f(np.arange(n))
        res["methods"][name] = {"uses": desc, "time_s": round(time.time() - t, 1),
                                "bootstrap_ari": round(bootstrap_ari(f, n, mc["bootstrap"] // 2, mc["bootstrap_frac"], seed), 3)}
        print(name, res["methods"][name], flush=True)
    for name, e in (("leiden_phys", e_phys), ("leiden_sim", e_sim)):
        t = time.time()
        lab, gamma = methods.leiden_k(e, ids, K, seed)
        labels[name] = lab
        res["methods"][name] = {"uses": "только сеть: " + ("физическая" if name.endswith("phys") else "сходство"),
                                "time_s": round(time.time() - t, 1), "resolution": round(float(gamma), 4),
                                "n_clusters": int(len(set(lab))), "bootstrap_ari": None}
    t = time.time()
    try:
        tuned = out / "canus_tuning.json"
        cp = dict(json.loads(tuned.read_text())["chosen"]["params"]) if tuned.exists() else {"epochs": mc["canus_epochs"]}
        ep = cp.pop("epochs", mc["canus_epochs"])
        labels["canus_phys"] = methods.canus(Y, methods.diffusion(Wp, cfg["network"]["diffusion_steps"]), K, seed, ep, **cp)
        res["methods"]["canus_phys"] = {"uses": "признаки + физическая сеть (код автора)", "time_s": round(time.time() - t, 1),
                                        "bootstrap_ari": None, "params": {"epochs": ep, **cp},
                                        "tuned": tuned.exists()}
    except Exception as ex:  # CANUS опционален: фиксируем причину
        res["methods"]["canus_phys"] = {"error": repr(ex)[:300]}
    print("canus", res["methods"]["canus_phys"], flush=True)

    ref = labels["kmeans"]
    for name, lab in labels.items():
        lab = methods.align_labels(ref, np.asarray(lab)) if len(set(lab)) == K else np.asarray(lab)
        labels[name] = lab
        ind = icvi.all_indices(Y, {"phys": Wp, "sim": Ws}, lab)
        rb = icvi.random_baseline(Y, {"phys": Wp, "sim": Ws}, lab, cfg["icvi"]["random_baseline"], seed)
        z = {k: (ind[k] - rb[k]["mean"]) / rb[k]["std"] if rb[k]["std"] > 0 else np.nan for k in ind}
        coh = agglo.coherence(pd.Series(lab, index=ids), u)
        r = res["methods"][name]
        r.update({"indices": {k: round(float(v), 4) for k, v in ind.items()},
                  "z_vs_random": {k: round(float(v), 1) for k, v in z.items()},
                  "ring_same_type": round(coh["ring"], 3), "control_same_type": round(coh["control"], 3),
                  "ari_regions": round(adjusted_rand_score(region, lab), 3),
                  "ari_vs_kmeans": round(adjusted_rand_score(ref, lab), 3),
                  "sizes": sorted(np.bincount(lab).tolist(), reverse=True)})
    # Ранги по шести индексам конкурса (MQ = Q; на физической сети)
    names = [m for m in res["methods"] if "indices" in res["methods"][m]]
    tab = pd.DataFrame({m: {k: res["methods"][m]["indices"][k if k in ("SW", "CH", "S_Dbw") else f"{k}@phys"] for k in CONTEST}
                        for m in names}).T
    ranks = pd.DataFrame({k: tab[k].rank(ascending=not icvi.HIGHER_BETTER[k]) for k in CONTEST})
    for m in names:
        res["methods"][m]["mean_rank_contest_icvi"] = round(float(ranks.loc[m].mean()), 2)
    (out / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    lab_df = pd.DataFrame(labels, index=pd.Index(ids, name="territory_id"))
    lab_df.insert(0, "type", lab_df["kefrin_phys"])
    lab_df.to_csv(out / "labels_static.csv")
    show = pd.DataFrame({m: {**{k: res["methods"][m]["indices"][k if k in ("SW", "CH", "S_Dbw") else f"{k}@phys"] for k in CONTEST},
                             "rank": res["methods"][m]["mean_rank_contest_icvi"], "ring": res["methods"][m]["ring_same_type"],
                             "ctrl": res["methods"][m]["control_same_type"], "ARIreg": res["methods"][m]["ari_regions"],
                             "boot": res["methods"][m].get("bootstrap_ari"), "ARIkm": res["methods"][m]["ari_vs_kmeans"]}
                         for m in names}).T
    print(show.round(3).to_string())


if __name__ == "__main__":
    main()
