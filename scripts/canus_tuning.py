"""Этап 3b, E6: честная настройка CANUS (метод Шалилеха, код автора) перед сравнением методов.

На этапе 2 CANUS запускался с настройками автора по умолчанию (100 эпох) и оказался последним. Чтобы сравнение было
честным, перебираем небольшую сетку параметров автора (configs/base.yaml: methods.canus_grid) на тех же данных
(годовой профиль 2024, диффузия физической сети) и берем вариант с лучшим средним рангом по шести ICVI конкурса
(SW, CH, S_Dbw по признакам; AVI, AVU, Q на физической сети) внутри сетки. Выбранные параметры использует
scripts/run_compare.py.
Выход: outputs/stage2/canus_tuning.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, icvi, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402

CONTEST = ["SW", "CH", "S_Dbw", "AVI", "AVU", "Q"]


def main() -> None:
    cfg = load()
    seed = cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage2"
    K = json.loads((out / "select_k.json").read_text())["chosen_K"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    Y = features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids].to_numpy()
    Wp = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    A = methods.diffusion(Wp, cfg["network"]["diffusion_steps"])
    rows = []
    for k, params in enumerate(cfg["methods"]["canus_grid"]):
        t = time.time()
        p = dict(params)
        ep = p.pop("epochs", cfg["methods"]["canus_epochs"])
        try:
            lab = methods.canus(Y, A, K, seed, ep, **p)
            ind = icvi.all_indices(Y, Wp, lab)
            row = {"k": k, "params": params, "time_s": round(time.time() - t, 1), "n_clusters": int(len(set(lab))),
                   **{c: round(float(ind[c]), 4) for c in CONTEST}}
        except Exception as ex:  # вариант не сошелся — фиксируем причину
            row = {"k": k, "params": params, "error": repr(ex)[:200]}
        rows.append(row)
        print(row, flush=True)
    tab = pd.DataFrame([r for r in rows if "error" not in r and r["n_clusters"] == K]).set_index("k")
    ranks = pd.DataFrame({c: tab[c].rank(ascending=not icvi.HIGHER_BETTER[c]) for c in CONTEST})
    tab["mean_rank"] = ranks.mean(axis=1)
    best = int(tab["mean_rank"].idxmin())
    res = {"definition": __doc__.split("Выход")[0].strip(), "K": K, "grid": rows,
           "mean_rank": {int(k): round(float(v), 2) for k, v in tab["mean_rank"].items()},
           "chosen": {"k": best, "params": rows[best]["params"]}}
    (out / "canus_tuning.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("выбран вариант", best, rows[best]["params"])


if __name__ == "__main__":
    main()
