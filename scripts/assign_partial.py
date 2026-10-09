"""Пропуски: что заполняем и что нет.

Покрытие данных о тратах по регионам: МО справочника, в данных, с полной панелью, с неполными рядами, без данных.
МО с неполными рядами в кластеризации не участвуют. Если у МО есть не меньше partial.min_months месяцев года профиля,
ему присваивается тип по ближайшему центру (src/mo/interpret.py: nearest_type) в тех же 7 признаках; если до
ближайшего центра дальше, чем у доли partial.atypical_quantile МО панели, — «нетипичный профиль».
Проверка правила на МО панели: прячем часть месяцев (случайные, половины года, только прошлый год) и сравниваем
тип по ближайшему центру с типом KEFRiN.
Почему остальное не заполняем: тип большинства соседей по границе и прогноз типа по показателям Росстата
(случайный лес, кросс-валидация) на МО панели — с какой долей они угадывают настоящий тип.
Выход: outputs/stage3/partial_types.csv, outputs/stage3/partial_types.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import data, features, interpret, panel  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    pc, seed = cfg["partial"], cfg["seed"]
    proc, out = path("processed"), path("outputs") / "stage3"
    year = cfg["features"]["profile_year"]
    prev = str(int(year) - 1)
    w = pd.read_parquet(proc / "panel_wide.parquet")
    ids = sorted(w.index.get_level_values("territory_id").unique())
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    code_of = interpret.type_names(lab)["code"].to_dict()
    pr = pd.read_parquet(proc / "profile.parquet").loc[ids]
    med = features.median_log_level(w)
    res = {"definition": __doc__.split("Выход")[0].strip()}

    # Покрытие: действующий справочник против данных о тратах
    b = data.borders()
    lat = data.borders_latest(b)
    lat = lat[lat["year_to"] == 9999]
    any_name = b.drop_duplicates("territory_id", keep="last").set_index("territory_id")
    in_data = set(int(t) for t in data.consumption()["territory_id"].unique())
    wp = panel.wide_partial()
    part = sorted(int(t) for t in wp.index.get_level_values("territory_id").unique())
    reg = lat["region_name"]
    cov = pd.DataFrame({"ref": reg.groupby(reg).size(),
                        "in_data": reg[reg.index.isin(in_data)].groupby(reg).size(),
                        "full": reg[reg.index.isin(ids)].groupby(reg).size(),
                        "partial": reg[reg.index.isin(part)].groupby(reg).size()}).fillna(0).astype(int)
    cov["no_data"] = cov["ref"] - cov["in_data"]
    absent = cov.index[cov["in_data"] == 0].tolist()
    partial_only = cov.index[(cov["in_data"] > 0) & (cov["full"] == 0)].tolist()
    res["coverage"] = {"mo_ref": int(cov["ref"].sum()), "mo_in_data": len(in_data), "mo_full": len(ids), "mo_partial": len(part),
                       "mo_no_data": int(cov["no_data"].sum()), "regions_ref": int(len(cov)),
                       "regions_full": int((cov["full"] > 0).sum()),
                       "absent_regions": absent, "absent_regions_mo": int(cov.loc[absent, "ref"].sum()),
                       "partial_only_regions": {r: {"ref": int(cov.loc[r, "ref"]), "partial": int(cov.loc[r, "partial"])}
                                                for r in partial_only}}

    # Проверка правила на МО панели: тип по ближайшему центру при части месяцев
    dates = w.index.get_level_values("date")
    in_year = dates.str.startswith(year)

    def agree(mask, yr=year) -> float:
        P = features.profile_partial(w[mask], med, yr)
        return round(float((interpret.nearest_type(P, pr, lab)["label"] == lab.reindex(P.index)).mean()), 3)

    rng = np.random.default_rng(seed)
    val = {"all_months": agree(in_year), "random_months": {}}
    for k in pc["validation_months"]:
        accs = []
        for _ in range(pc["validation_reps"]):
            r = pd.Series(rng.random(len(w)), index=w.index).where(in_year)
            accs.append(agree((r.groupby(level="territory_id").rank() <= k).to_numpy()))
        val["random_months"][str(k)] = round(float(np.mean(accs)), 3)
    months = sorted(set(dates[in_year]))
    val["first_half"] = agree(dates.isin(months[:len(months) // 2]))
    val["second_half"] = agree(dates.isin(months[len(months) // 2:]))
    val["prev_year_only"] = agree(dates.str.startswith(prev), prev)
    res["validation"] = val

    # МО с неполными рядами: тип по ближайшему центру, если месяцев года профиля достаточно
    wq = wp[wp.index.get_level_values("date").str.startswith(year)]
    n_year = wq.groupby(level="territory_id").size().reindex(part).fillna(0).astype(int)
    n_all = wp.groupby(level="territory_id").size().reindex(part).astype(int)
    ok = n_year.index[n_year >= pc["min_months"]]
    thr = float(interpret.nearest_type(pr, pr, lab)["distance"].quantile(pc["atypical_quantile"]))
    nt = interpret.nearest_type(features.profile_partial(wp.loc[list(ok)], med, year), pr, lab)
    status = pd.Series("few_months", index=part)
    status.loc[ok] = np.where(nt["distance"] > thr, "atypical", "type")
    name = lambda t: (lat.loc[t, "municipal_district_name"] if t in lat.index else any_name.loc[t, "municipal_district_name"])  # noqa: E731
    region = lambda t: (lat.loc[t, "region_name"] if t in lat.index else any_name.loc[t, "region_name"])  # noqa: E731
    rows = pd.DataFrame({"territory_id": part, "name": [name(t) for t in part], "region": [region(t) for t in part],
                         "months_total": n_all.to_numpy(), f"months_{year}": n_year.to_numpy(), "status": status.to_numpy()})
    rows["nearest"] = rows["territory_id"].map(nt["label"].map(code_of)).fillna("")
    rows["type"] = np.where(rows["status"] == "type", rows["nearest"], "")
    rows["distance"] = rows["territory_id"].map(nt["distance"].round(2))
    rows.to_csv(out / "partial_types.csv", index=False)
    typed, atyp = rows[rows["status"] == "type"], rows[rows["status"] == "atypical"]
    res["rule"] = {"min_months": pc["min_months"], "atypical_quantile": pc["atypical_quantile"],
                   "atypical_distance": round(thr, 2), "year": year}
    res["result"] = {"eligible": int(len(ok)), "typed": int(len(typed)), "atypical": int(len(atyp)),
                     "few_months": int((rows["status"] == "few_months").sum()),
                     "typed_by_type": {k: int(v) for k, v in typed["type"].value_counts().sort_index().items()},
                     "typed_by_region": {k: int(v) for k, v in typed["region"].value_counts().items()},
                     "atypical_by_region": {k: int(v) for k, v in atyp["region"].value_counts().items()},
                     "atypical_full_year": int((atyp[f"months_{year}"] == 12).sum()),
                     "months_total_le_12": int((rows["months_total"] <= 12).sum()),
                     "partial_only_regions_typed": {r: int((typed["region"] == r).sum()) for r in partial_only}}

    # Почему остальное не заполняем: соседи по границе и прогноз по Росстату (на МО панели, где ответ известен)
    mo = pd.read_parquet(proc / "mo.parquet").loc[ids]
    fed = mo["type"].str.startswith("внутригородская")
    cont = pd.read_parquet(proc / "edges_contiguity.parquet")
    pos = set(ids)
    nb: dict = {}
    for a, c in zip(cont["a"], cont["b"]):
        if a in pos and c in pos:
            nb.setdefault(a, []).append(c)
            nb.setdefault(c, []).append(a)

    def neighbours(keep) -> float:
        hits = []
        for t, ns in nb.items():
            if keep(t):
                cnt = Counter(lab[n] for n in ns).most_common()
                top = [k for k, v in cnt if v == cnt[0][1]]
                hits.append(len(top) == 1 and top[0] == lab[t])
        return round(float(np.mean(hits)), 3)

    X = mo[pc["rosstat_features"]].astype(float)
    for col in pc["rosstat_log"]:
        X[col] = np.log(X[col])
    okx = X.notna().all(axis=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def rosstat(keep) -> float:
        m = okx & keep
        rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=1)
        return round(float(cross_val_score(rf, X[m], lab[m], cv=cv).mean()), 3)

    res["not_filled"] = {"neighbours_majority": neighbours(lambda t: True),
                         "neighbours_majority_no_fed_cities": neighbours(lambda t: not fed[t]),
                         "n_with_neighbours": len(nb),
                         "rosstat_forest": rosstat(pd.Series(True, index=ids)),
                         "rosstat_forest_no_fed_cities": rosstat(~fed),
                         "rosstat_features": pc["rosstat_features"],
                         "base_rate": round(float(lab.value_counts(normalize=True).max()), 3),
                         "base_rate_no_fed_cities": round(float(lab[~fed].value_counts(normalize=True).max()), 3)}
    (out / "partial_types.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
