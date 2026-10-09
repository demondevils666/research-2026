"""Этап 2, шаг 5: динамика типов по 24 месяцам (эволюционный KEFRiN) и выбор β.

Для каждого β из configs/base.yaml (dynamics.betas): метки по месяцам, средний силуэт срезов, доля сменивших тип,
доля МО без смен, ARI соседних месяцев, ARI со статической типологией, события. β = 0 — без сглаживания (только теплый
старт). Выбор β*: минимум доли смен при потере среднего силуэта не более dynamics.sw_loss_max от β = 0.
База сравнения: независимая кластеризация каждого месяца + выравнивание.
Выход: outputs/stage2/dynamics.json, outputs/stage2/labels_monthly.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, dynamics, methods, networks  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    dc, seed = cfg["dynamics"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage2"
    K = json.loads((out / "select_k.json").read_text())["chosen_K"]
    alpha = json.loads((out / "kefrin_sweep.json").read_text())["chosen_alpha"]
    static_df = pd.read_csv(out / "labels_static.csv", index_col="territory_id")
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    static = static_df.loc[ids, "type"].to_numpy()
    P = methods.network_matrix(networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids))
    Ym = dynamics.per_month_standardized(pd.read_parquet(proc / "features_monthly.parquet"))
    u = agglo.units(mo, pd.read_parquet(proc / "ring_pairs.parquet"), pd.read_parquet(proc / "edges_contiguity.parquet"))

    res = {"K": K, "alpha": alpha, "rule": __doc__.split("Выход")[0].strip(), "by_beta": {}}
    runs = {}
    for b in dc["betas"]:
        r = dynamics.evolve(Ym, P, static, K, alpha, b, seed)
        runs[b] = r
        s = dynamics.summary(r["labels"], r["sw_month"], static, K, dc["event_min_months"])
        res["by_beta"][str(b)] = {k: v for k, v in s.items() if k not in ("events", "changed_share_by_month", "transition_matrix")}
        print(b, {k: round(v, 4) if isinstance(v, float) else v for k, v in res["by_beta"][str(b)].items()}, flush=True)
    ind = dynamics.independent(Ym, P, static, K, alpha, seed, cfg["methods"]["n_init"])
    s_ind = dynamics.summary(ind["labels"], ind["sw_month"], static, K, dc["event_min_months"])
    res["independent_monthly"] = {k: v for k, v in s_ind.items() if k not in ("events", "changed_share_by_month", "transition_matrix")}
    print("independent", res["independent_monthly"], flush=True)

    sw0 = res["by_beta"]["0.0"]["sw_month_mean"]
    ok = [b for b in dc["betas"] if res["by_beta"][str(b)]["sw_month_mean"] >= (1 - dc["sw_loss_max"]) * sw0]
    best = min(ok, key=lambda b: (res["by_beta"][str(b)]["changed_share_mean"], b))
    res["chosen_beta"] = best
    final = dynamics.summary(runs[best]["labels"], runs[best]["sw_month"], static, K, dc["event_min_months"])
    names = mo["name"]
    final["events"] = [{"territory_id": int(ids[i]), "name": names[ids[i]], "month": d, "from": a, "to": c}
                       for i, d, a, c in final["events"]]
    res["final"] = final
    # Агломерации во времени: доля «бубликов» в типе своего города по месяцам
    res["final"]["ring_same_type_by_month"] = {
        d: round(agglo.coherence(pd.Series(runs[best]["labels"][d], index=ids), u)["ring"], 3) for d in sorted(Ym)}
    (out / "dynamics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    pd.DataFrame({d: runs[best]["labels"][d] for d in sorted(Ym)}, index=pd.Index(ids, name="territory_id")).to_csv(
        out / "labels_monthly.csv")
    print("chosen beta:", best, {k: final[k] for k in ("sw_month_mean", "changed_share_mean", "never_changed_share",
                                                       "ari_consecutive_mean", "ari_with_static_mean", "n_events")})


if __name__ == "__main__":
    main()
