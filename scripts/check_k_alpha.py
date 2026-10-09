"""Этап 4: проверка числа типов K при итоговом весе сети α.

K выбирался до развертки α, при α = methods.kefrin_alpha (scripts/select_k.py), а итоговый α выбрала развертка
(scripts/run_kefrin_sweep.py). Здесь та же процедура выбора K повторяется для KEFRiN при итоговом α: 20 подвыборок по 80%
МО, медианный ARI с полным разбиением и минимальный по типам Жаккар Хеннига; то же правило (наибольшее устойчивое K).
Выход: outputs/stage2/k_at_alpha.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from mo import features, icvi, networks  # noqa: E402
from mo.config import load, path  # noqa: E402
from select_k import is_stable, stability, subsamples  # noqa: E402


def main() -> None:
    cfg = load()
    mc, seed = cfg["methods"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage2"
    alpha = json.loads((out / "kefrin_sweep.json").read_text())["chosen_alpha"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    W = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    K_final = json.loads((out / "select_k.json").read_text())["chosen_K"]
    final = pd.read_csv(path("outputs") / "stage3" / "mo_types.csv", index_col="territory_id").loc[ids, "type"].to_numpy()
    subs = subsamples(len(ids), mc, seed)
    rows = []
    for K in mc["k_grid"]:
        full, ari_med, _, jac = stability("kefrin", Y, W, K, alpha, subs, seed, mc["n_init"])
        ind = icvi.all_indices(Y, W, full)
        rows.append({"K": K, "ari_median": round(ari_med, 3), "jaccard_min": round(float(jac.min()), 3),
                     **{k: round(float(ind[k]), 4) for k in ("SW", "CH", "S_Dbw", "AVI", "AVU", "Q")}})
        if K == K_final:
            # Жаккар по типам итоговой типологии: кластер подгонки → тип, с которым он больше всего пересекается
            ct = pd.crosstab(full, final)
            rows[-1]["jaccard_by_type"] = {ct.loc[c].idxmax(): round(float(jac[i]), 3) for i, c in enumerate(sorted(ct.index))}
        print(rows[-1], flush=True)
    ok = [r["K"] for r in rows if is_stable(r["ari_median"], r["jaccard_min"], mc)]
    res = {"definition": __doc__.split("Выход")[0].strip(), "alpha": alpha, "rows": rows, "stable_K": ok,
           "chosen_K": max(ok) if ok else None,
           "chosen_K_at_selection_alpha": json.loads((out / "select_k.json").read_text())["chosen_K"]}
    (out / "k_at_alpha.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("стабильные K:", ok, "выбор:", res["chosen_K"])


if __name__ == "__main__":
    main()
