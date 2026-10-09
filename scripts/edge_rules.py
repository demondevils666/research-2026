"""Этап 4: как правило ребра меняет кластеризацию — KEFRiN на всех шести сетях этапа 2.

Условие конкурса просит показать влияние правила ребра на кластеризацию. Для каждой сети (configs/base.yaml:
edge_rules.networks) повторяем главный эксперимент (scripts/run_kefrin_sweep.py): тот же K, тот же перебор веса сети α
и то же правило выбора α (максимум AVI на своей сети при потере силуэта не больше sweep.sw_loss_max, ARI с регионами
не выше sweep.region_ari_max, без мелких типов и с той же устойчивостью, что при выборе K: бутстрап-ARI не ниже
methods.stability_ari_min и Жаккар каждого типа не ниже methods.stability_jaccard_min).
Физическая сеть берется из главного эксперимента (outputs/stage2/kefrin_sweep.json), остальные считаются здесь.
Чтобы не считать бутстрап для всех α, сначала для каждого α считаются дешевые условия правила, а бутстрап — по кандидатам
в порядке убывания AVI, пока один не пройдет порог устойчивости. Выбранный α тот же, что при полном переборе.
На выбранном α для каждой сети:
  SW, CH, S_Dbw — компактность по тратам;
  AVI и Q на своей сети и на физической (общий масштаб для сравнения сетей);
  ARI с итоговой типологией и с регионами, доля «бубликов» в типе своего города против контроля, бутстрап-ARI.
Выход: outputs/stage2/edge_rules.json, data/processed/labels_edge_rules.parquet
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


def candidates(df: pd.DataFrame, cfg: dict, n: int) -> pd.DataFrame:
    """α, проходящие дешевые условия правила (силуэт, регионы, размер типов), по убыванию AVI на своей сети."""
    sw0 = df.loc[0, "SW"]
    min_size = df["sizes"].apply(min) / n
    ok = df[(df["SW"] >= (1 - cfg["sweep"]["sw_loss_max"]) * sw0) & (df["ari_regions"] <= cfg["sweep"]["region_ari_max"])
            & (min_size >= cfg["sweep"]["min_cluster_share"])]
    return ok.sort_values(["AVI_own", "alpha"], ascending=[False, True])


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "stage2"
    K = json.loads((out / "select_k.json").read_text())["chosen_K"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    Wp = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    u = agglo.units(mo, pd.read_parquet(proc / "ring_pairs.parquet"), pd.read_parquet(proc / "edges_contiguity.parquet"))
    region = mo.loc[ids, "region_code"].to_numpy()
    final = pd.read_csv(out / "labels_static.csv", index_col="territory_id").loc[ids, "type"].to_numpy()
    rules = json.loads((out / "networks.json").read_text())["rules"]
    mc = cfg["methods"]
    subs = subsamples(len(ids), mc, cfg["seed"])          # та же проверка устойчивости, что при выборе K и α
    res = {"definition": __doc__.split("Выход")[0].strip(), "K": K, "networks": {}}
    labs = {}
    # физическая сеть: результаты главного эксперимента
    sw = json.loads((out / "kefrin_sweep.json").read_text())
    a_phys = sw["chosen_alpha"]
    lab_phys = pd.read_parquet(proc / "labels_sweep.parquet").loc[ids, f"a{a_phys:.2f}"].to_numpy()
    for name in cfg["edge_rules"]["networks"]:
        W = Wp if name == "physical" else networks.to_dense(pd.read_parquet(proc / f"net_{name}.parquet"), ids)
        def boot(a, lab_a):
            _, ari_med, _, jac = stability("kefrin", Y, W, K, a, subs, cfg["seed"], mc["n_init"], full=lab_a)
            return round(ari_med, 3), round(float(jac.min()), 3)
        if name == "physical":
            rows = [{**r, "AVI_own": r["AVI"]} for r in sw["rows"]]
            best = next(r for r in rows if r["alpha"] == a_phys)
            lab = lab_phys
        else:
            P = methods.network_matrix(W)
            rows, by_a, base = [], {}, None
            for a in cfg["sweep"]["alphas"]:
                lab_a, _ = methods.kefrin(Y, W, K, a, cfg["seed"], cfg["methods"]["n_init"], Z=methods.kefrin_Z(Y, P, a))
                base = lab_a if base is None else base
                lab_a = methods.align_labels(base, lab_a)
                by_a[a] = lab_a
                ind = icvi.all_indices(Y, W, lab_a)
                rows.append({"alpha": a, "SW": round(float(ind["SW"]), 4), "AVI_own": round(float(ind["AVI"]), 4),
                             "ari_regions": round(adjusted_rand_score(region, lab_a), 3), "bootstrap_ari": None,
                             "sizes": np.bincount(lab_a, minlength=K).tolist()})
                print(name, rows[-1], flush=True)
            best = None
            for _, c in candidates(pd.DataFrame(rows), cfg, len(ids)).iterrows():
                r = next(r for r in rows if r["alpha"] == c["alpha"])
                r["bootstrap_ari"], r["jaccard_min"] = boot(r["alpha"], by_a[r["alpha"]])
                print(name, "устойчивость", r["alpha"], r["bootstrap_ari"], r["jaccard_min"], flush=True)
                if is_stable(r["bootstrap_ari"], r["jaccard_min"], mc):
                    best = r
                    break
            if best is None:                                   # ни один α не прошел: только признаки
                best = rows[0]
                if best["bootstrap_ari"] is None:
                    best["bootstrap_ari"], best["jaccard_min"] = boot(best["alpha"], by_a[best["alpha"]])
            lab = by_a[best["alpha"]]
        lab = methods.align_labels(final, lab)
        labs[name] = lab
        ind_own = icvi.all_indices(Y, W, lab)
        ind_phys = icvi.all_indices(Y, Wp, lab)
        coh = agglo.coherence(pd.Series(lab, index=ids), u)
        res["networks"][name] = {
            "rule": rules[name], "alpha": float(best["alpha"]),
            "admissible": bool(best["bootstrap_ari"] >= cfg["methods"]["stability_ari_min"]),
            "SW": round(float(ind_own["SW"]), 4), "CH": round(float(ind_own["CH"]), 1), "S_Dbw": round(float(ind_own["S_Dbw"]), 4),
            "AVI_own": round(float(ind_own["AVI"]), 4), "Q_own": round(float(ind_own["Q"]), 4),
            "AVI_phys": round(float(ind_phys["AVI"]), 4), "Q_phys": round(float(ind_phys["Q"]), 4),
            "ari_final": round(adjusted_rand_score(final, lab), 3), "ari_regions": round(adjusted_rand_score(region, lab), 3),
            "ring_same_type": round(coh["ring"], 3), "control_same_type": round(coh["control"], 3),
            "bootstrap_ari": float(best["bootstrap_ari"]), "sizes": np.bincount(lab, minlength=K).tolist(),
            "sweep": [{k: r[k] for k in ("alpha", "SW", "AVI_own", "ari_regions", "bootstrap_ari")} for r in rows],
        }
        print(name, {k: v for k, v in res["networks"][name].items() if k not in ("sweep", "rule")}, flush=True)
    (out / "edge_rules.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    pd.DataFrame(labs, index=pd.Index(ids, name="territory_id")).to_parquet(proc / "labels_edge_rules.parquet")


if __name__ == "__main__":
    main()
