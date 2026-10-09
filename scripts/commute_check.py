"""Этап 4: прямая проверка механизма «живут в пригороде, работают в городе» по переписи 2010 года.

Механизм главной находки: Росстат считает зарплату по месту работы, СберИндекс — траты по месту жительства, поэтому
район, чьи жители работают в городе, тратит как город. Данных о поездках на работу у СберИндекса нет; их дает
Всероссийская перепись населения 2010 года (региональные издания, том 7: занятые по территории нахождения работы
по городским округам и муниципальным районам; разбор — src/mo/census.py, регионы — configs/base.yaml: commute).
Мера: доля занятых, работающих в другом населенном пункте своего региона. Поездка в соседнее село или в свой райцентр
тоже считается, а уровень сильно различается по регионам, поэтому сравниваем с типичным районом своего региона:
отклонение от медианы муниципальных районов региона по переписи (городские округа в медиану не входят).
Группы МО (outputs/stage3/satellites.csv): «бублики»; скрытые спутники (обычные соседи города, у которых индекс спутника
по тратам не ниже медианы «бубликов»); прочие соседи города (контроль); остальные МО региона — фон.
Перепись в индекс спутника не входит, поэтому для скрытых спутников это независимая проверка.
Тесты: разность средних отклонений «бублики против прочих соседей», «спутники против прочих соседей» — перестановки
меток внутри региона (commute.n_perm); Спирмен индекса спутника с отклонением — по всем соседям города и только по
обычным соседям (без «бубликов», чей статус задан справочником). Промахи: «бублики» и спутники с отклонением ниже нуля;
прочие соседи с отклонением не ниже медианы «бубликов».
Проверка меры: доля работающих вне своего населенного пункта вообще (включая другой регион и другие страны).
Разведочно (не заранее заданный тест): те же группы отдельно для районов со своим городом (доля горожан Росстата не ниже
половины) и без него.
Ограничения: 2010 год против трат 2023–2024; границы части МО с 2010 года менялись (сопоставление по названиям);
регионы — те, где таблица опубликована по районам; в районе со своим городом его жители работают у себя, и мера
по району в целом «размывается».
Выход: outputs/stage3/commute_check.json, outputs/stage3/commute_check.csv
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import census, stats  # noqa: E402
from mo.config import load, path  # noqa: E402

GROUPS = ("ring", "hidden", "control", "other")
# основная мера и проверка: доля работающих вне своего населенного пункта, включая другой регион и другие страны
MEASURES = ("share_other_settlement", "share_outside_settlement")


def perm_within(dev: pd.Series, group: pd.Series, region: pd.Series, a: str, b: str, n: int, seed: int) -> dict:
    """Разность средних отклонений групп a и b; p — доля перестановок меток внутри региона с не меньшей разностью
    по модулю (двусторонний тест)."""
    m = group.isin([a, b])
    d, g, r = dev[m].to_numpy(), group[m].to_numpy(), region[m].to_numpy()
    obs = d[g == a].mean() - d[g == b].mean()
    rng = np.random.default_rng(seed)
    idx = [np.where(r == x)[0] for x in np.unique(r)]
    cnt = 0
    for _ in range(n):
        gp = g.copy()
        for i in idx:
            gp[i] = g[i][rng.permutation(len(i))]
        cnt += abs(d[gp == a].mean() - d[gp == b].mean()) >= abs(obs) - 1e-12
    return {"diff_pp": round(float(obs) * 100, 1), "p_value": round(float((cnt + 1) / (n + 1)), 4), "n_perm": n,
            "n_a": int((g == a).sum()), "n_b": int((g == b).sum())}


def spearman(x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    r, p = spearmanr(x[ok], y[ok])
    return {"rho": round(float(r), 3), "p_value": round(float(p), 4), "n": int(ok.sum())}


def main() -> None:
    cfg = load()
    cc = cfg["commute"]
    raw, proc, out = path("raw"), path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    sat = pd.read_csv(out / "satellites.csv", index_col="territory_id")
    rows, regions = [], {}
    for reg, spec in cc["regions"].items():
        df = census.read_region(raw, spec)
        hit, unmatched, missing = census.match(df, mo.loc[mo["region"] == reg, "name"], spec.get("aliases"))
        # район по переписи 2010 (часть МО с тех пор стала городским или муниципальным округом); если вид МО в названии
        # не указан («Абзелиловский» в таблице Башкортостана), вид берется из справочника
        dir_kind = {u: census.kind(mo.loc[t, "name"]) for u, t in zip(hit["unit"], hit["territory_id"])}
        df["district_2010"] = [census.kind(u) == "district" or (census.kind(u) == "?" and dir_kind.get(u) == "district")
                               for u in df["unit"]]
        med = {m: float(df.loc[df["district_2010"], m].median()) for m in MEASURES}
        hit = hit.assign(region=reg, region_median=med[MEASURES[0]], region_median_outside=med[MEASURES[1]],
                         district_2010=hit["unit"].map(dict(zip(df["unit"], df["district_2010"]))))
        rows.append(hit)
        regions[reg] = {"file": spec["file"], "n_census_units": len(df), "n_districts_2010": int(df["district_2010"].sum()),
                        "district_median_pct": round(med[MEASURES[0]] * 100, 1), "n_matched": len(hit),
                        "n_directory": int((mo["region"] == reg).sum()),
                        "census_not_in_data": unmatched, "data_not_in_census": missing}
        print(reg, len(df), "->", len(hit), "из", regions[reg]["n_directory"], "| медиана районов",
              round(med[MEASURES[0]] * 100, 1), "%")
    t = pd.concat(rows).set_index("territory_id")
    t = t[t["share_other_settlement"].notna()]                    # скрытые ячейки («K») в итоговых строках
    t["dev"] = t["share_other_settlement"] - t["region_median"]
    t["dev_outside"] = t["share_outside_settlement"] - t["region_median_outside"]
    t["name"] = mo.loc[t.index, "name"]
    s = sat.reindex(t.index)
    t["group"] = np.where(s["group"] == "ring", "ring",
                          np.where(s["group"] == "control", np.where(s["hidden"] == True, "hidden", "control"),  # noqa: E712
                                   "other"))
    t["index"], t["city"] = s["index"], s["city"]

    summary = {}
    for g in GROUPS:
        x = t[t["group"] == g]
        summary[g] = {"n": len(x), "mean_dev_pp": round(float(x["dev"].mean()) * 100, 1) if len(x) else None,
                      "mean_dev_outside_pp": round(float(x["dev_outside"].mean()) * 100, 1) if len(x) else None,
                      "median_dev_pp": round(float(x["dev"].median()) * 100, 1) if len(x) else None,
                      "median_share_pct": round(float(x["share_other_settlement"].median()) * 100, 1) if len(x) else None,
                      "share_above_region_median": round(float((x["dev"] > 0).mean()), 3) if len(x) else None,
                      "mean_dev_ci95_pp": [round(v * 100, 1) for v in stats.bootstrap_ci(x["dev"], n=2000, seed=cfg["seed"])]
                      if len(x) > 1 else None}
    n, seed = cc["n_perm"], cfg["seed"]
    tests = {"ring_vs_control": perm_within(t["dev"], t["group"], t["region"], "ring", "control", n, seed),
             "hidden_vs_control": perm_within(t["dev"], t["group"], t["region"], "hidden", "control", n, seed),
             "ring_vs_control_outside": perm_within(t["dev_outside"], t["group"], t["region"], "ring", "control", n, seed),
             "hidden_vs_control_outside": perm_within(t["dev_outside"], t["group"], t["region"], "hidden", "control", n, seed)}
    nb = t[t["group"].isin(["ring", "hidden", "control"])]
    corr = {"index_all_neighbours": spearman(nb["index"], nb["dev"]),
            "index_ordinary_neighbours": spearman(nb.loc[nb["group"] != "ring", "index"],
                                                  nb.loc[nb["group"] != "ring", "dev"])}
    # Разведочный разрез (не заранее заданный тест): мера переписи по МО «размывается» жителями своего города района,
    # которые работают у себя. Делим районы по доле горожан Росстата (не ниже половины — «со своим городом»).
    town = mo.loc[t.index, "urban_share"] >= 0.5
    explore = {g: {k: {"n": int(m.sum()), "mean_dev_pp": round(float(t.loc[m, "dev"].mean()) * 100, 1) if m.any() else None}
                   for k, m in (("own_town", (t["group"] == g) & town), ("rural", (t["group"] == g) & ~town))}
               for g in ("ring", "hidden", "control")}
    # Разведочный разрез по столице региона (как в пробе P1: у партнера-города статус «центр_субъекта»): держится ли
    # разница «бублик против обычного соседа» и у нестоличных городов, где по профилю трат она незначима
    cap = s["partner"].map(mo["status"].astype(str).str.contains("центр_субъекта")).fillna(False).astype(bool)
    by_capital = {}
    for k, m in (("capital", cap), ("not_capital", ~cap)):
        x = t[m]
        by_capital[k] = {g: {"n": int((x["group"] == g).sum()),
                             "mean_dev_pp": round(float(x.loc[x["group"] == g, "dev"].mean()) * 100, 1)}
                         for g in ("ring", "control")}
        by_capital[k]["ring_vs_control"] = perm_within(x["dev"], x["group"], x["region"], "ring", "control", n, seed)
    ring_med = float(t.loc[t["group"] == "ring", "dev"].median())
    cols = ["name", "region", "group", "share_other_settlement", "dev", "index", "city"]

    def recs(x: pd.DataFrame) -> list[dict]:
        x = x[cols].assign(share_pct=(x["share_other_settlement"] * 100).round(1), dev_pp=(x["dev"] * 100).round(1),
                           index=x["index"].round(2))
        return x.drop(columns=["share_other_settlement", "dev"]).reset_index().to_dict("records")

    misses = {"flagged_below_region_median": recs(t[t["group"].isin(["ring", "hidden"]) & (t["dev"] < 0)].sort_values("dev")),
              "control_at_ring_level": recs(t[(t["group"] == "control") & (t["dev"] >= ring_med)].sort_values("dev", ascending=False))}
    # Витрина (commute.showcase): район вокруг города глазами Росстата, трат и переписи
    sc = cc["showcase"]
    names = mo.loc[mo["region"] == sc["region"], "name"]
    key = {census.norm_name(v): k for k, v in names.items()}
    cid, rid = key[census.norm_name(sc["city"])], key[census.norm_name(sc["district"])]
    w = pd.read_parquet(proc / "panel_wide.parquet")["Все категории"]
    year = cfg["features"]["profile_year"]
    spend = w[w.index.get_level_values("date").str.startswith(year)].groupby(level="territory_id").mean()
    types = pd.read_csv(out / "mo_types.csv", index_col="territory_id")["type"]
    showcase = {"region": sc["region"], "city": mo.loc[cid, "name"], "district": mo.loc[rid, "name"], "city_id": int(cid),
                "district_id": int(rid),
                "year": year, "district_population": int(mo.loc[rid, "population"]),
                "district_rural_pct": round((1 - float(mo.loc[rid, "urban_share"])) * 100, 0),
                "wage_pct_of_city": round(float(mo.loc[rid, "wage"] / mo.loc[cid, "wage"]) * 100, 0),
                "spend_pct_of_city": round(float(spend[rid] / spend[cid]) * 100, 0),
                "commute_pct": round(float(t.loc[rid, "share_other_settlement"]) * 100, 1),
                "region_median_pct": round(float(t.loc[rid, "region_median"]) * 100, 1),
                "district_type": types[rid], "city_type": types[cid], "district_group": t.loc[rid, "group"]}
    res = {"definition": __doc__.split("Выход")[0].strip(), "regions": regions, "n_matched": len(t), "showcase": showcase,
           "groups": summary, "tests": tests, "spearman": corr, "ring_median_dev_pp": round(ring_med * 100, 1),
           "exploratory_by_own_town": explore, "exploratory_by_partner_capital": by_capital,
           "misses": misses,
           "top": recs(t[t["group"] != "other"].sort_values("dev", ascending=False).head(10))}
    (out / "commute_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    keep = ["region", "name", "unit", "district_2010", "group", "index", "city", "employed", "own_region", "own_settlement",
            "other_region", "unknown", "share_other_settlement", "region_median", "dev"]
    t[[c for c in keep if c in t]].round(4).sort_values(["region", "name"]).to_csv(out / "commute_check.csv")
    print(json.dumps({k: res[k] for k in ("groups", "tests", "spearman")}, ensure_ascii=False, indent=1))
    for k, v in misses.items():
        print(k, [(m["name"], m["share_pct"], m["dev_pp"], m["index"]) for m in v])


if __name__ == "__main__":
    main()
