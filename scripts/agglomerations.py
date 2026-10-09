"""Этап 3, шаг 2: «город больше своих границ» — формальные проверки и масштаб.

Пары «бублик» (район вокруг города с общим центром) — город; контроль — обычные районы, граничащие с городским
округом своего региона (src/mo/agglo.py). Три независимых измерения:
  1) профиль трат: ближе ли район к своему городу, чем к районам своего региона (проба P1, outputs/probes/p1_*.json);
  2) синхронность: корреляция помесячных изменений относительного уровня трат район—город (23 изменения),
     сравнение с контролем (перестановочный тест) и со случайными парами «район — город» из других регионов;
  3) место работы против места жительства: связь отношения трат район/город с отношением зарплат (Росстат, по месту
     работы) у «бубликов» и у контроля, бутстрап-ДИ и перестановочный тест разницы.
Масштаб: сколько людей живет в «бубликах» (Росстат 2023, все 146 пар справочника) и какая у них доля горожан;
куда итоговая типология относит «бублики» по сравнению с их городами.
Выход: outputs/stage3/agglomerations.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, data, interpret, links, rosstat, stats  # noqa: E402
from mo.config import load, path  # noqa: E402


def corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    za = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    zb = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    return (za * zb).mean(1)


def main() -> None:
    cfg = load()
    seed, npm, nb = cfg["seed"], cfg["probes"]["permutations"], cfg["probes"]["bootstrap"]
    proc, out = path("processed"), path("outputs") / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    u = agglo.units(mo, rp, pd.read_parquet(proc / "edges_contiguity.parquet"))
    res = {"definition": __doc__.split("Выход")[0].strip()}

    # 1) Профиль трат — из пробы P1 (тот же код, этап 1)
    p1 = json.loads((path("outputs") / "probes" / "p1_ring_pairs.json").read_text())
    res["profile_closeness"] = {k: p1["main"][k] for k in ("n_ring", "n_control", "share_closer_ring", "share_closer_ring_ci95",
                                                            "share_closer_control", "share_closer_control_ci95", "perm_share_diff")}
    res["profile_closeness"]["by_city_status"] = {
        "region_capital": {k: p1["partner_is_region_capital"][k] for k in ("n_ring", "share_closer_ring", "share_closer_control")},
        "other_city": {k: p1["partner_not_region_capital"][k] for k in ("n_ring", "share_closer_ring", "share_closer_control")}}
    # Чувствительность: контроль из всех соседних городов, включая города других регионов
    res["profile_closeness"]["control_any_region"] = {k: p1["control_any_region"][k] for k in ("n_control", "share_closer_control")}

    # 2) Синхронность помесячных изменений уровня трат
    lvl = pd.read_parquet(proc / "features_monthly.parquet")["rel_level"].unstack("date").loc[ids]
    dl = lvl.diff(axis=1).iloc[:, 1:]
    corr = {}
    for g in ("ring", "control"):
        uu = u[u["group"] == g]
        corr[g] = corr_rows(dl.loc[uu["unit"]].to_numpy(), dl.loc[uu["partner"]].to_numpy())
    # Случайные пары: «бублик» с чужим городом-центром из другого региона (сохраняем типы участников)
    rng = np.random.default_rng(seed)
    cities = u.loc[u["group"] == "ring", "partner"].to_numpy()
    rings = u.loc[u["group"] == "ring", "unit"].to_numpy()
    reg = mo["region_code"]
    rand = []
    for r in rings:
        pool = [c for c in cities if reg[c] != reg[r]]
        for c in rng.choice(pool, 20, replace=False):
            rand.append((r, c))
    rand = np.array(rand)
    corr["random_other_region"] = corr_rows(dl.loc[rand[:, 0]].to_numpy(), dl.loc[rand[:, 1]].to_numpy())
    res["comovement"] = {
        "series": "помесячные изменения относительного уровня трат (23 значения), корреляция Пирсона",
        **{f"mean_corr_{g}": round(float(v.mean()), 3) for g, v in corr.items()},
        **{f"median_corr_{g}": round(float(np.median(v)), 3) for g, v in corr.items()},
        "ring_ci95_mean": [round(x, 3) for x in stats.bootstrap_ci(corr["ring"], n=nb, seed=seed)],
        "control_ci95_mean": [round(x, 3) for x in stats.bootstrap_ci(corr["control"], n=nb, seed=seed)],
        "perm_ring_vs_control": stats.perm_test_diff(corr["ring"], corr["control"], n=npm, seed=seed),
        "perm_ring_vs_random": stats.perm_test_diff(corr["ring"], corr["random_other_region"], n=npm, seed=seed),
        "share_corr_gt_0_5": {g: round(float((v > 0.5).mean()), 3) for g, v in corr.items()},
    }
    # Взаимная «синхронная восьмерка» в сети со-движения (этап 2)
    e = pd.read_parquet(proc / "net_corr.parquet")
    E = set(zip(e["a"], e["b"]))
    cov = {g: float(np.mean([(min(a, b), max(a, b)) in E for a, b in zip(uu["unit"], uu["partner"])]))
           for g, uu in u.groupby("group")}
    k = cfg["network"]["knn_k"]
    cov["random_expected"] = 1 - (1 - k / (len(ids) - 1)) ** 2  # в kNN-объединении: i в top-k у j или j у i
    res["comovement"]["share_pairs_in_corr_knn"] = {g: round(v, 3) for g, v in cov.items()}

    # 3) Место работы против места жительства
    w = pd.read_parquet(proc / "panel_wide.parquet")
    total = cfg["panel"]["total_category"]
    wy = w[w.index.get_level_values("date").str.startswith(cfg["features"]["profile_year"])].groupby(level="territory_id").mean()
    rows = {}
    for g, uu in u.groupby("group"):
        ls = np.log(wy.loc[uu["unit"], total].to_numpy() / wy.loc[uu["partner"], total].to_numpy())
        lw = np.log(mo.loc[uu["unit"], "wage"].to_numpy() / mo.loc[uu["partner"], "wage"].to_numpy())
        ok = np.isfinite(ls) & np.isfinite(lw)
        rows[g] = (ls[ok], lw[ok])
    def corr2(x):
        return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    mw = {}
    for g, (ls, lw) in rows.items():
        X = np.column_stack([ls, lw])
        bs = np.random.default_rng(seed)
        boot = [corr2(X[bs.integers(0, len(X), len(X))]) for _ in range(nb)]
        mw[g] = {"n": int(len(X)), "corr_spend_ratio_vs_wage_ratio": round(corr2(X), 3),
                 "ci95": [round(float(np.quantile(boot, 0.025)), 3), round(float(np.quantile(boot, 0.975)), 3)],
                 # Уровни: медианные траты и зарплата района в % к своему городу (exp медианы логарифма отношения)
                 "median_spend_pct": round(float(np.exp(np.median(ls)) * 100), 1),
                 "median_wage_pct": round(float(np.exp(np.median(lw)) * 100), 1),
                 "share_wage_below_80pct": round(float((lw < np.log(0.8)).mean()), 3),
                 # Как у Иркутского района: траты (в % к городу) выше зарплаты (в % к городу), в том числе на 20+ п. п.
                 "share_spend_above_wage": round(float((ls > lw).mean()), 3),
                 "share_spend_above_wage_20pp": round(float((np.exp(ls) - np.exp(lw) >= 0.2).mean()), 3)}
    # Перестановочный тест разницы корреляций: перемешиваем принадлежность пар к группам
    Xr, Xc = np.column_stack(rows["ring"]), np.column_stack(rows["control"])
    pool = np.vstack([Xr, Xc])
    obs = corr2(Xr) - corr2(Xc)
    prng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(npm):
        prng.shuffle(pool)
        cnt += abs(corr2(pool[:len(Xr)]) - corr2(pool[len(Xr):])) >= abs(obs)
    mw["perm_diff_ring_minus_control"] = {"diff": round(obs, 3), "p_value": round((cnt + 1) / (npm + 1), 4)}
    mw["interpretation"] = ("Росстат считает зарплату по месту работы (организации на территории МО), СберИндекс — траты "
                            "жителей МО. Если жители района работают в городе, местная зарплата не связана с их тратами.")
    res["workplace_vs_residence"] = mw

    # Масштаб: все пары справочника (146), население Росстата 2023
    b = data.borders()
    rp_all = links.ring_pairs(b[b["year_to"] == 9999])
    pop, _ = rosstat.population(b, cfg["rosstat"]["year"])
    rpop = pop.reindex(rp_all["ring"])
    cpop = pop.reindex(rp_all["city"])
    res["scale"] = {
        "ring_pairs_total": int(len(rp_all)), "ring_pairs_in_panel": int(len(rp)),
        "ring_population_mln": round(float(rpop["population"].sum()) / 1e6, 2),
        "ring_population_known_n": int(rpop["population"].notna().sum()),
        "ring_urban_share_weighted": round(float((rpop["population"] * rpop["urban_share"]).sum()
                                                 / rpop.loc[rpop["urban_share"].notna(), "population"].sum()), 3),
        "ring_rural_population_mln": round(float((rpop["population"] * (1 - rpop["urban_share"])).sum()) / 1e6, 2),
        "city_population_mln": round(float(cpop["population"].sum()) / 1e6, 2),
        "ring_to_city_population_ratio_median": round(float(np.nanmedian(rpop["population"].to_numpy()
                                                                         / cpop["population"].to_numpy())), 3),
    }
    # Где итоговая типология держит «бублики» по сравнению с городами
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"]
    names = interpret.type_names(lab.loc[ids])
    code = lab.map(names["code"])
    ct = pd.crosstab(code.loc[rp["city"]].to_numpy(), code.loc[rp["ring"]].to_numpy(), rownames=["city"], colnames=["ring"])
    res["typology"] = {
        "ring_same_type_as_city": round(float((code.loc[rp["ring"]].to_numpy() == code.loc[rp["city"]].to_numpy()).mean()), 3),
        "crosstab_city_by_ring": {r: {c: int(v) for c, v in row.items() if v} for r, row in ct.iterrows()},
        "ring_types": code.loc[rp["ring"]].value_counts().to_dict(),
        "control_types": code.loc[u.loc[u["group"] == "control", "unit"]].value_counts().to_dict(),
    }
    (out / "agglomerations.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
