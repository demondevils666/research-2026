"""Этап 3, шаг 3: пространство и динамика итоговых типов.

Пространство: «пояс или архипелаг» на графе соседства по границе для итоговых типов (lift и z против случайной
перестановки; доля МО типа в его крупнейшем связном куске) — со всеми МО и без внутригородских территорий Москвы и СПб.
Динамика (эволюционный KEFRiN, outputs/stage2/labels_monthly.csv): вероятность остаться в типе за месяц,
модальный тип МО, «пограничные» МО (в модальном типе меньше 75% месяцев), устойчивые смены (события) по парам типов.
Отслеживание самих типов (MONIC): внешние переходы между соседними месяцами (выживание, расщепление, поглощение,
исчезновение, рождение) и внутренние — размер типа по месяцам и медианные доли категорий у тех же МО по годам.
Выход: outputs/stage3/space_dynamics.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import dynamics, interpret, methods  # noqa: E402
from mo.config import load, path  # noqa: E402
from mo.spatial import assortativity, type_stats  # noqa: E402


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    names = interpret.type_names(lab)
    code_of = names["code"].to_dict()
    name_of = names.set_index("code")["name"].to_dict()
    cont = pd.read_parquet(proc / "edges_contiguity.parquet")
    res = {"definition": __doc__.split("Выход")[0].strip()}

    # Пространство
    pos = {t: i for i, t in enumerate(ids)}
    L = lab.to_numpy()
    au, av = cont["a"].map(pos).to_numpy(), cont["b"].map(pos).to_numpy()
    keep = ~mo.loc[ids, "type"].str.startswith("внутригородская").to_numpy()
    ok = keep[au] & keep[av]
    remap = -np.ones(len(ids), dtype=int)
    remap[keep] = np.arange(keep.sum())
    n_perm = cfg["probes"]["permutations"] // 4
    ts_all = type_stats(L, au, av, n_perm, cfg["seed"])
    ts_nf = type_stats(L[keep], remap[au[ok]], remap[av[ok]], n_perm, cfg["seed"])
    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(zip(cont["a"], cont["b"]))
    space = {}
    for k in sorted(set(L)):
        c = code_of[k]
        members = [t for t in ids if lab[t] == k]
        comps = sorted((len(x) for x in nx.connected_components(g.subgraph(members))), reverse=True)
        r_all = ts_all[ts_all["type"] == k].iloc[0]
        r_nf = ts_nf[ts_nf["type"] == k].iloc[0]
        space[c] = {"name": name_of[c], "n": int(len(members)),
                    "lift_all": float(r_all["lift"]), "z_all": float(r_all["z"]),
                    "lift_no_fed_cities": float(r_nf["lift"]), "z_no_fed_cities": float(r_nf["z"]),
                    "same_edge_share_no_fed": float(r_nf["same_edge_share"]), "expected_no_fed": float(r_nf["expected_random"]),
                    "largest_patch_share": round(comps[0] / len(members), 3), "n_patches": len(comps),
                    "has_same_type_neighbor": float(r_all["has_same_neighbor"])}
    res["space"] = {"types": space, "assortativity_all": round(assortativity(L, au, av), 3),
                    "assortativity_no_fed_cities": round(assortativity(L[keep], remap[au[ok]], remap[av[ok]]), 3)}
    # Те же меры для типологии без сети (α = 0): пояса и архипелаг — от поведения или от подмешанной географии?
    L0 = methods.align_labels(L, pd.read_parquet(proc / "labels_sweep.parquet").loc[ids, "a0.00"].to_numpy())
    ts0 = type_stats(L0, au, av, n_perm, cfg["seed"])
    ts0_nf = type_stats(L0[keep], remap[au[ok]], remap[av[ok]], n_perm, cfg["seed"])
    res["space"]["alpha0"] = {
        "types": {code_of[k]: {"lift_all": float(ts0[ts0["type"] == k].iloc[0]["lift"]),
                               "lift_no_fed_cities": float(ts0_nf[ts0_nf["type"] == k].iloc[0]["lift"])} for k in sorted(set(L0))},
        "assortativity_all": round(assortativity(L0, au, av), 3),
        "assortativity_no_fed_cities": round(assortativity(L0[keep], remap[au[ok]], remap[av[ok]]), 3)}

    # Динамика
    dyn = json.loads((path("outputs") / "stage2" / "dynamics.json").read_text())
    LM = pd.read_csv(path("outputs") / "stage2" / "labels_monthly.csv", index_col="territory_id").loc[ids]
    T = np.array(dyn["final"]["transition_matrix"])
    modal = LM.mode(axis=1)[0].astype(int)
    modal_share = (LM.to_numpy() == modal.to_numpy()[:, None]).mean(1)
    boundary = pd.Series(modal_share < 0.75, index=ids)
    ev = dyn["final"]["events"]
    pairs = Counter((code_of[e["from"]], code_of[e["to"]]) for e in ev)
    per_mo_events = Counter(e["territory_id"] for e in ev)
    big = mo.loc[ids].sort_values("population", ascending=False)
    examples = []
    for t in big.index:
        if per_mo_events.get(t, 0) and len(examples) < 12:
            seq = LM.loc[t].map(code_of)
            examples.append({"name": mo.loc[t, "name"], "population": int(mo.loc[t, "population"]),
                             "static_type": code_of[lab[t]], "path": " → ".join(seq.loc[seq.ne(seq.shift())].tolist())})
    # Устойчивость типа отдельно для районов Москвы и Петербурга и для остальных МО
    M = LM.to_numpy()
    same = M[:, 1:] == M[:, :-1]
    fed = ~keep
    stay_split = {code_of[k]: {"no_fed_cities": round(float(same[(M[:, :-1] == k) & ~fed[:, None]].mean()), 3),
                               "fed_cities": (round(float(same[(M[:, :-1] == k) & fed[:, None]].mean()), 3)
                                              if ((M[:, :-1] == k) & fed[:, None]).any() else None),
                               "n_fed_cities_static": int(((L == k) & fed).sum())}
                  for k in sorted(set(L))}
    # Сезонность смен типа: смены внутри 2023 и 2024 года в одни и те же месяцы у одного МО против случайного совпадения
    ch = M[:, 1:] != M[:, :-1]
    m = len(LM.columns) // 2 - 1                          # переходов внутри года (янв→фев … ноя→дек)
    c23, c24 = ch[:, :m], ch[:, m + 1:2 * m + 1]
    same_month = int((c23 & c24).sum())
    expected = float((c23.sum(1) * c24.sum(1) / m).sum())
    repeat = (M[:, :m + 1] == M[:, m + 1:]).all(1) & (c23.sum(1) > 0)
    seasonal = {"changes_by_transition": ch.sum(0).astype(int).tolist(),
                "transitions": [f"{a}→{b}" for a, b in zip(LM.columns[:-1], LM.columns[1:])],
                "same_month_both_years": same_month, "expected_if_random": round(expected, 1),
                "agreement_by_lag": {int(g): round(float((M[:, g:] == M[:, :-g]).mean()), 3) for g in (1, 3, 6, 9, 12)},
                "n_mo_repeating_year_cycle": int(repeat.sum()),
                "examples_repeating_cycle": [mo.loc[t, "name"] for t in big.index if repeat[ids.index(t)]][:8]}
    res["dynamics"] = {
        "beta": dyn["chosen_beta"],
        "stay_probability": {code_of[k]: round(float(T[k, k]), 3) for k in range(len(T))},
        "stay_probability_split": stay_split,
        "seasonality": seasonal,
        "changed_share_mean": round(dyn["final"]["changed_share_mean"], 4),
        "never_changed_share": round(dyn["final"]["never_changed_share"], 3),
        "modal_equals_static_share": round(float((modal.to_numpy() == L).mean()), 3),
        "boundary_mo_share": round(float(boundary.mean()), 3),
        "boundary_share_by_static_type": {code_of[k]: round(float(boundary[lab == k].mean()), 3) for k in sorted(set(L))},
        "n_events": len(ev),
        "events_by_pair_top": [{"from": a, "to": b, "n": n} for (a, b), n in pairs.most_common(10)],
        "examples_large_mo_with_events": examples,
        "ring_same_type_by_month": dyn["final"]["ring_same_type_by_month"],
    }
    # Отслеживание типов (MONIC). Внешние переходы: тип месяца t и типы месяца t+1 по доле общих МО
    mc = cfg["dynamics"]["monic"]
    steps = [dynamics.monic(M[:, i], M[:, i + 1], mc["match"], mc["split"]) for i in range(M.shape[1] - 1)]
    kinds = ("survive", "split", "absorb", "disappear", "emerge")
    ks = sorted(set(L), key=lambda k: code_of[k])             # в порядке T1…T6
    over = {code_of[k]: [s["overlap"][k] for s in steps if k in s["overlap"]] for k in ks}
    # Внутренние переходы: размер типа по месяцам и профиль тех же МО (статический тип) в каждом году
    w = pd.read_parquet(proc / "panel_wide.parquet")
    total = cfg["panel"]["total_category"]
    cats = [c for c in w.columns if c != total]
    years = sorted({d[:4] for d in w.index.get_level_values("date")})
    prof = {}
    for y in years:
        wy = w[w.index.get_level_values("date").str.startswith(y)].groupby(level="territory_id").mean().loc[ids]
        sh = wy[cats].div(wy[total], axis=0)
        prof[y] = {code_of[k]: {c: round(float(sh.loc[lab == k, c].median()), 4) for c in cats} for k in ks}
    res["dynamics"]["monic"] = {
        "thresholds": mc, "n_steps": len(steps),
        "events": {kd: int(sum(len(s[kd]) for s in steps)) for kd in kinds},
        "survive_same_label": int(sum(sum(1 for x, y in s["survive"].items() if x == y) for s in steps)),
        "overlap_by_type": {c: {"min": round(min(v), 3), "median": round(float(np.median(v)), 3)} for c, v in over.items()},
        "overlap_min": round(min(min(v) for v in over.values()), 3),
        "size_by_type": {code_of[k]: {"static": int((L == k).sum()), "min": int((M == k).sum(0).min()),
                                      "max": int((M == k).sum(0).max())} for k in ks},
        "profile_by_year": prof,
        "transition_matrix": {"order": [code_of[k] for k in ks],
                              "matrix": [[round(float(T[a, b]), 3) for b in ks] for a in ks]},   # как stay_probability
    }
    (out / "space_dynamics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(pd.DataFrame(space).T.to_string())
    print(json.dumps({k: v for k, v in res["dynamics"].items() if k not in ("ring_same_type_by_month",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
