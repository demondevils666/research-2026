"""Этап 2, главный эксперимент: как вес физической сети α в KEFRiN меняет типологию.

Для каждого α (configs/base.yaml: sweep.alphas) при K из select_k.json:
  SW, CH, S_Dbw — компактность по признакам (траты);
  AVI, AVU, Q — связность на физической сети;
  связность агломераций — доля «бубликов» в типе своего города против контроля (обычные районы у города своего региона);
  ARI с регионами — насколько типы превращаются в регионы; ARI с α = 0 — насколько типология сдвинулась.
Выбор α*: максимум AVI на физической сети при потере SW не более sweep.sw_loss_max, ARI с регионами не выше
sweep.region_ari_max, без кластеров меньше sweep.min_cluster_share и с той же устойчивостью, что требует выбор K
(scripts/select_k.py: те же подвыборки 80%, медианный ARI >= methods.stability_ari_min и минимальный покластерный
Жаккар >= methods.stability_jaccard_min). Связность агломераций не входит в правило выбора (чтобы не подгонять под
гипотезу), а показывается как следствие.
Выход: outputs/stage2/kefrin_sweep.json, data/processed/labels_sweep.parquet
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, features, icvi, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402
from select_k import is_stable, stability, subsamples  # noqa: E402


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "stage2"
    K = json.loads((out / "select_k.json").read_text())["chosen_K"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    W = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    u = agglo.units(mo, pd.read_parquet(proc / "ring_pairs.parquet"), pd.read_parquet(proc / "edges_contiguity.parquet"))
    region = mo.loc[ids, "region_code"].to_numpy()
    rows, labs = [], {}
    base = None
    P = methods.network_matrix(W)  # диффузия считается один раз
    mc = cfg["methods"]
    subs = subsamples(len(ids), mc, cfg["seed"])
    for a in cfg["sweep"]["alphas"]:
        lab, _ = methods.kefrin(Y, W, K, a, cfg["seed"], mc["n_init"], Z=methods.kefrin_Z(Y, P, a))
        _, ari_med, _, jac = stability("kefrin", Y, W, K, a, subs, cfg["seed"], mc["n_init"], full=lab)
        if base is None:
            base = lab
        lab = methods.align_labels(base, lab)
        labs[f"a{a:.2f}"] = lab
        s = pd.Series(lab, index=ids)
        coh = agglo.coherence(s, u)
        ind = icvi.all_indices(Y, W, lab)
        rows.append({"alpha": a, **{k: round(float(ind[k]), 4) for k in ["SW", "CH", "S_Dbw", "AVI", "AVU", "Q"]},
                     "ring_same_type": round(coh["ring"], 3), "control_same_type": round(coh["control"], 3),
                     "ari_regions": round(adjusted_rand_score(region, lab), 3),
                     "ari_vs_alpha0": round(adjusted_rand_score(base, lab), 3),
                     "bootstrap_ari": round(ari_med, 3), "jaccard_min": round(float(jac.min()), 3),
                     "stable": is_stable(ari_med, float(jac.min()), mc),
                     "sizes": np.bincount(lab, minlength=K).tolist()})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    sw0 = df.loc[0, "SW"]
    min_size = df["sizes"].apply(min) / len(ids)
    ok = df[(df["SW"] >= (1 - cfg["sweep"]["sw_loss_max"]) * sw0) & (df["ari_regions"] <= cfg["sweep"]["region_ari_max"])
            & (min_size >= cfg["sweep"]["min_cluster_share"]) & df["stable"]]
    best = ok.sort_values(["AVI", "alpha"], ascending=[False, True]).iloc[0]
    res = {"K": K, "rule": __doc__.split("Выход")[0].strip(), "rows": rows, "chosen_alpha": float(best["alpha"]),
           "admissible_alphas": ok["alpha"].tolist()}
    (out / "kefrin_sweep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    pd.DataFrame(labs, index=pd.Index(ids, name="territory_id")).to_parquet(proc / "labels_sweep.parquet")
    print(df.drop(columns="sizes").to_string(index=False))
    print("chosen alpha:", res["chosen_alpha"], "admissible:", res["admissible_alphas"])


if __name__ == "__main__":
    main()
