"""Этап 3b, E2: сеть по месяцам и качество типов во времени.

Динамическая атрибутированная сеть G_t = (V, E, W_t, X_t): узлы и ребра — физическая сеть (граница × расстояние,
не зависит от трат), вес ребра в месяце t умножается на сходство соседей в этом месяце:
W_t(i, j) = w_phys(i, j) · exp(−‖x_i(t) − x_j(t)‖² / 2h²) (src/mo/networks.py: monthly_layer).
Что считаем по месяцам:
  средний вес и среднее сходство соседей (сближаются ли соседи), устойчивость весов (ранговая корреляция весов
  ребер соседних месяцев), доля «граничных» ребер между разными типами (помесячные типы эволюционного KEFRiN);
  сходство «бублик — город» против «обычный район — город» (держится ли находка каждый месяц);
  шесть ICVI конкурса по месяцам (SW, CH, S_Dbw по признакам месяца; AVI, AVU, Q на физической сети и на W_t)
  против случайной перестановки меток;
  эволюционный KEFRiN на W_t (те же α*, β*) против основного варианта с постоянной сетью: ARI по месяцам, доля смен.
Выход: outputs/stage3/monthly_network.json, outputs/stage3/monthly_network.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, dynamics, icvi, methods, networks, stats  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    mc, seed = cfg["monthly_network"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage3"
    s2 = path("outputs") / "stage2"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    K = json.loads((s2 / "select_k.json").read_text())["chosen_K"]
    alpha = json.loads((s2 / "kefrin_sweep.json").read_text())["chosen_alpha"]
    beta = json.loads((s2 / "dynamics.json").read_text())["chosen_beta"]
    static = pd.read_csv(s2 / "labels_static.csv", index_col="territory_id").loc[ids, "type"].to_numpy()
    LM = pd.read_csv(s2 / "labels_monthly.csv", index_col="territory_id").loc[ids]
    phys = pd.read_parquet(proc / "net_physical.parquet")
    Ym = dynamics.per_month_standardized(pd.read_parquet(proc / "features_monthly.parquet"))
    dates = sorted(Ym)
    layers, h = networks.monthly_layer(phys, ids, Ym, mc["h"])
    W0 = networks.to_dense(phys, ids)
    pos = {t: i for i, t in enumerate(ids)}
    ia, ib = phys["a"].map(pos).to_numpy(), phys["b"].map(pos).to_numpy()
    res = {"definition": __doc__.split("Выход")[0].strip(), "h": round(h, 4), "K": K, "alpha": alpha, "beta": beta,
           "n_edges": int(len(phys))}

    # Пары «бублик — город» и «обычный район — город», которые являются ребрами физической сети
    u = agglo.units(mo, pd.read_parquet(proc / "ring_pairs.parquet"), pd.read_parquet(proc / "edges_contiguity.parquet"))
    key = {(min(a, b), max(a, b)): k for k, (a, b) in enumerate(zip(phys["a"], phys["b"]))}
    u["edge"] = [key.get((min(a, b), max(a, b))) for a, b in zip(u["unit"], u["partner"])]
    ue = u.dropna(subset=["edge"]).astype({"edge": int})

    rows, base_rows, rng_seed = [], [], seed
    prev = None
    for d in dates:
        L = layers[d]
        lab = LM[d].to_numpy()
        boundary = lab[ia] != lab[ib]
        sim_ring = L["sim"].to_numpy()[ue.loc[ue["group"] == "ring", "edge"]]
        sim_ctrl = L["sim"].to_numpy()[ue.loc[ue["group"] == "control", "edge"]]
        Wt = networks.to_dense(L, ids)
        ind = icvi.all_indices(Ym[d], {"phys": W0, "month": Wt}, lab)
        row = {"date": d, "mean_w": float(L["w"].mean()), "mean_sim": float(L["sim"].mean()),
               "boundary_edge_share": float(boundary.mean()),
               "boundary_weight_share": float(L["w"].to_numpy()[boundary].sum() / L["w"].sum()),
               "sim_ring_city": float(sim_ring.mean()), "sim_control_city": float(sim_ctrl.mean()),
               "w_rank_corr_prev": (float(spearmanr(prev, L["w"].to_numpy())[0]) if prev is not None else np.nan)}
        for k in ("SW", "CH", "S_Dbw", "AVI@phys", "AVU@phys", "Q@phys", "AVI@month", "AVU@month", "Q@month"):
            row[k] = float(ind[k])
        rows.append(row)
        base = icvi.random_baseline(Ym[d], {"phys": W0, "month": Wt}, lab, mc["baseline_permutations"], rng_seed)
        base_rows.append({"date": d, **{f"{k}_z": (ind[k] - base[k]["mean"]) / base[k]["std"] if base[k]["std"] > 0 else np.nan
                                         for k in ("SW", "CH", "S_Dbw", "AVI@phys", "AVU@phys", "Q@phys", "AVI@month",
                                                   "AVU@month", "Q@month")}})
        prev = L["w"].to_numpy()
        print(d, {k: round(v, 3) for k, v in row.items() if k != "date"}, flush=True)
    tab = pd.DataFrame(rows).set_index("date").join(pd.DataFrame(base_rows).set_index("date"))
    tab.round(5).to_csv(out / "monthly_network.csv")

    n = mc["trend_window"]
    first, last = tab.iloc[:n], tab.iloc[-n:]
    res["neighbors_similarity"] = {
        "mean_sim_first_months": round(float(first["mean_sim"].mean()), 4),
        "mean_sim_last_months": round(float(last["mean_sim"].mean()), 4),
        "spearman_with_time": round(float(spearmanr(np.arange(len(tab)), tab["mean_sim"])[0]), 3),
        "w_rank_corr_consecutive_mean": round(float(tab["w_rank_corr_prev"].mean()), 3),
        "w_rank_corr_first_last": round(float(spearmanr(layers[dates[0]]["w"], layers[dates[-1]]["w"])[0]), 3),
        "boundary_edge_share_mean": round(float(tab["boundary_edge_share"].mean()), 3),
        "boundary_edge_share_range": [round(float(tab["boundary_edge_share"].min()), 3),
                                      round(float(tab["boundary_edge_share"].max()), 3)],
    }
    res["ring_vs_control_by_month"] = {
        "n_ring_edges": int((ue["group"] == "ring").sum()), "n_control_edges": int((ue["group"] == "control").sum()),
        "sim_ring_mean": round(float(tab["sim_ring_city"].mean()), 4),
        "sim_control_mean": round(float(tab["sim_control_city"].mean()), 4),
        "months_ring_gt_control": int((tab["sim_ring_city"] > tab["sim_control_city"]).sum()), "months": len(tab),
    }
    res["icvi_by_month"] = {
        k: {"mean": round(float(tab[k].mean()), 4), "min": round(float(tab[k].min()), 4), "max": round(float(tab[k].max()), 4),
            "z_mean": round(float(tab[f"{k}_z"].mean()), 1), "z_min": round(float(tab[f"{k}_z"].min()), 1)}
        for k in ("SW", "CH", "S_Dbw", "AVI@phys", "AVU@phys", "Q@phys", "AVI@month", "AVU@month", "Q@month")}

    # Эволюционный KEFRiN на сети по месяцам против основного (постоянная сеть)
    Pm = {d: methods.network_matrix(networks.to_dense(layers[d], ids)) for d in dates}
    r = dynamics.evolve(Ym, Pm, static, K, alpha, beta, seed)
    sm = dynamics.summary(r["labels"], r["sw_month"], static, K, cfg["dynamics"]["event_min_months"])
    main_sum = json.loads((s2 / "dynamics.json").read_text())["final"]
    ari = [adjusted_rand_score(LM[d].to_numpy(), r["labels"][d]) for d in dates]
    same = [float((LM[d].to_numpy() == r["labels"][d]).mean()) for d in dates]
    res["evolutionary_kefrin_monthly_network"] = {
        "ari_with_main_by_month_mean": round(float(np.mean(ari)), 3), "ari_with_main_min": round(float(np.min(ari)), 3),
        "same_type_share_mean": round(float(np.mean(same)), 3),
        "changed_share_mean": round(float(sm["changed_share_mean"]), 4),
        "changed_share_mean_main": round(float(main_sum["changed_share_mean"]), 4),
        "sw_month_mean": round(float(sm["sw_month_mean"]), 4), "sw_month_mean_main": round(float(main_sum["sw_month_mean"]), 4),
        "never_changed_share": round(float(sm["never_changed_share"]), 3),
        "never_changed_share_main": round(float(main_sum["never_changed_share"]), 3),
    }
    # Тест на уровне пар: среднее за 24 месяца сходство каждой пары, «бублики» против обычных районов
    S = np.column_stack([layers[d]["sim"].to_numpy() for d in dates])
    pair_sim = pd.Series(S[ue["edge"].to_numpy()].mean(1), index=ue.index)
    res["ring_vs_control_by_month"]["perm_pairs_mean_sim"] = stats.perm_test_diff(
        pair_sim[ue["group"] == "ring"], pair_sim[ue["group"] == "control"], n=cfg["probes"]["permutations"], seed=seed)
    (out / "monthly_network.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
