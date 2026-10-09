"""Этап 4: «экономические двойники» МО и расхождения развития (сценарий Лаборатории СберИндекс).

Директор лаборатории о цели конкурса: поиск экономически сопоставимых муниципалитетов «позволяет оценивать потенциал,
искать применимый опыт и замечать расхождения развития».
Двойники: для каждого МО — k ближайших МО из ДРУГИХ регионов по профилю трат 2023 года (CLR шести долей +
относительный уровень, стандартизованы); внутригородские территории Москвы и Петербурга — двойники только друг другу.
Расхождение развития: рост трат на жителя 2023→2024, %, у МО против медианы роста у его двойников — с поправкой на рост
своего региона: (рост МО − медиана роста МО его региона) − медиана того же у двойников, п. п. Двойники из других
регионов, поэтому без поправки в разницу входит общий рост региона; его долю показываем отдельно.
Проверки: двойники чаще своего типа и ближе по зарплате Росстата, чем случайные МО из других регионов.
Выход: outputs/stage3/twins.json, outputs/stage3/twins.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, interpret, panel  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    tc, seed = cfg["twins"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    code = lab.map(interpret.type_names(lab)["code"])
    w = panel.wide()
    X = features.standardize(features.profile(w, tc["profile_year"]).loc[ids]).to_numpy()
    total = cfg["panel"]["total_category"]
    sp = {y: w[w.index.get_level_values("date").str.startswith(y)].groupby(level="territory_id").mean().loc[ids, total]
          for y in (tc["profile_year"], tc["next_year"])}
    growth = ((sp[tc["next_year"]] / sp[tc["profile_year"]] - 1) * 100).to_numpy()
    reg = mo.loc[ids, "region_code"].to_numpy()
    reg_growth = pd.Series(growth).groupby(reg).transform("median").to_numpy()   # медиана роста МО своего региона
    rel = growth - reg_growth
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    D[reg[:, None] == reg[None, :]] = np.inf          # двойники — только из других регионов
    # Один уровень МО (twins.same_level): району не подбираем в пару внутригородскую территорию Москвы или Петербурга
    intra = mo.loc[ids, "type"].astype(str).str.startswith("внутригородская").to_numpy()
    level = intra if tc.get("same_level") else np.zeros(len(ids), bool)
    D[level[:, None] != level[None, :]] = np.inf
    k = tc["k"]
    nn = np.argsort(D, axis=1)[:, :k]
    twin_growth = np.median(growth[nn], axis=1)
    gap_raw = growth - twin_growth
    div = rel - np.median(rel[nn], axis=1)              # расхождение с поправкой на рост своего региона, п. п.
    # Сколько разброса «МО минус двойники» без поправки объясняет регион: R² средних по регионам
    fe = pd.Series(gap_raw).groupby(reg).transform("mean").to_numpy()
    region_r2 = 1 - float(((gap_raw - fe) ** 2).sum() / ((gap_raw - gap_raw.mean()) ** 2).sum())
    # Проверки: тип и зарплата двойников против случайных МО из других регионов
    rng = np.random.default_rng(seed)
    rnd = np.array([rng.choice(np.where((reg != reg[i]) & (level == level[i]))[0], k, replace=False) for i in range(len(ids))])
    cd = code.to_numpy()
    lw = np.log(mo.loc[ids, "wage"].to_numpy())
    same_type_twins = float((cd[nn] == cd[:, None]).mean())
    same_type_rand = float((cd[rnd] == cd[:, None]).mean())
    wage_diff_twins = float(np.nanmedian(np.abs(lw[nn] - lw[:, None])))
    wage_diff_rand = float(np.nanmedian(np.abs(lw[rnd] - lw[:, None])))
    names = mo.loc[ids, "name"].to_numpy()
    regions = mo.loc[ids, "region"].to_numpy()
    # Пример регионального фона: регион (не меньше 10 МО), где медленнее двойников без поправки растет больше всего МО
    below = pd.DataFrame({"region": regions, "below": gap_raw < 0}).groupby("region")["below"].agg(["mean", "size"])
    below = below[below["size"] >= 10].sort_values("mean", ascending=False, kind="stable")
    rows = []
    for i, t in enumerate(ids):
        rows.append({"territory_id": t, "name": names[i], "region": regions[i], "type": cd[i],
                     "twins": ";".join(str(ids[j]) for j in nn[i]), "twin_dist": ";".join(f"{D[i, j]:.3f}" for j in nn[i]),
                     "growth_pct": round(float(growth[i]), 2), "twins_growth_pct": round(float(twin_growth[i]), 2),
                     "region_growth_pct": round(float(reg_growth[i]), 2), "gap_raw_pp": round(float(gap_raw[i]), 2),
                     "gap_adj_pp": round(float(div[i]), 2)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "twins.csv", index=False)
    order = np.argsort(div)

    def ex(ix):
        return [{"name": names[i], "region": regions[i], "type": cd[i], "gap": round(float(div[i]), 1),
                 "twins": [f"{names[j]} ({regions[j]})" for j in nn[i][:3]]} for ix_ in [ix] for i in ix_]

    big = mo.loc[ids, "population"].to_numpy() >= tc["examples_min_population"]
    res = {"definition": __doc__.split("Выход")[0].strip(), "k": k,
           "check": {"same_type_twins": round(same_type_twins, 3), "same_type_random_other_region": round(same_type_rand, 3),
                     "median_abs_log_wage_diff_twins": round(wage_diff_twins, 3),
                     "median_abs_log_wage_diff_random": round(wage_diff_rand, 3)},
           "gap_adj_sd_pp": round(float(div.std()), 2),
           "region_share_of_raw_gap_variance": round(region_r2, 3),
           "example_region": {"name": below.index[0], "share_below_twins": round(float(below["mean"].iloc[0]), 3),
                              "n": int(below["size"].iloc[0])},
           "top_ahead_of_twins": ex([i for i in order[::-1] if big[i]][:8]),
           "top_behind_twins": ex([i for i in order if big[i]][:8])}
    (out / "twins.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "definition"}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
