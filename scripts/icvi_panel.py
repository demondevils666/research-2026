"""Этап 4: панель ICVI по правилам отчета лаборатории жюри (Shalileh, Antonov, Tsyplakova, Doklady Mathematics 2025).

Правила из статьи (doi:10.1134/S1064562425700589):
  SW устойчив; CH растет с числом объектов N — приводим CH/N; уровень S_Dbw на случайном разбиении зависит от K —
  нормируем на случайные разбиения; AVI на случайном разбиении ≈ 1/K — показываем рядом с 1/K; AVU, ANUI, Q — по сети.
  Ориентиры «сильной» структуры: SW ≥ 0,5, CH/N ≥ 1, S_Dbw ≤ 0,6.
Для каждого метода (outputs/stage2/labels_static.csv): значения; z против перестановок меток с сохранением размеров
типов; 95%-интервалы по подвыборкам 80% МО (индексы сети — на индуцированном подграфе); отметки об ориентирах.
Выход: outputs/stage2/icvi_panel.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, icvi, networks  # noqa: E402
from mo.config import load, path  # noqa: E402

KEYS = ["SW", "CH_N", "S_Dbw", "AVI", "AVU", "ANUI", "Q"]


def panel(Y: np.ndarray, W: np.ndarray, lab: np.ndarray) -> dict:
    f = icvi.feature_indices(Y, lab)
    g = icvi.graph_indices(W, lab)
    return {"SW": f["SW"], "CH_N": f["CH"] / len(lab), "S_Dbw": f["S_Dbw"], "AVI": g["AVI"], "AVU": g["AVU"],
            "ANUI": g["ANUI"], "Q": g["Q"]}


def main() -> None:
    cfg = load()
    pc, seed = cfg["icvi_panel"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage2"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    W = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    L = pd.read_csv(out / "labels_static.csv", index_col="territory_id").loc[ids]
    rng = np.random.default_rng(seed)
    res = {"definition": __doc__.split("Выход")[0].strip(), "thresholds": pc["strong_structure"], "methods": {}}
    for m in [c for c in L.columns if c != "type"]:
        lab = L[m].to_numpy()
        K = len(set(lab))
        val = panel(Y, W, lab)
        perm = pd.DataFrame([panel(Y, W, rng.permutation(lab)) for _ in range(pc["permutations"])])
        sub = []
        for _ in range(pc["subsamples"]):
            s = np.sort(rng.choice(len(ids), int(pc["subsample_frac"] * len(ids)), replace=False))
            sub.append(panel(Y[s], W[np.ix_(s, s)], lab[s]))
        sub = pd.DataFrame(sub)
        th = pc["strong_structure"]
        res["methods"][m] = {
            "K": K, "inv_K": round(1 / K, 4),
            "value": {k: round(float(v), 4) for k, v in val.items()},
            "random_mean": {k: round(float(perm[k].mean()), 4) for k in KEYS},
            "z": {k: round(float((val[k] - perm[k].mean()) / perm[k].std(ddof=0)), 1) if perm[k].std(ddof=0) > 0 else None
                  for k in KEYS},
            "ci95_subsample": {k: [round(float(sub[k].quantile(0.025)), 4), round(float(sub[k].quantile(0.975)), 4)]
                               for k in KEYS},
            "meets_strong_structure": {"SW": bool(val["SW"] >= th["SW"]), "CH_N": bool(val["CH_N"] >= th["CH_N"]),
                                       "S_Dbw": bool(val["S_Dbw"] <= th["S_Dbw"])},
            "avi_over_inv_K": round(float(val["AVI"] * K), 2),
        }
        print(m, res["methods"][m]["value"], res["methods"][m]["z"], flush=True)
    (out / "icvi_panel.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
