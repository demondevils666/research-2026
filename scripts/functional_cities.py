"""Этап 3b, E1: «функциональные города» — какие районы живут как свой город, даже если справочник этого не знает.

Для каждой пары «район — город» (65 «бубликов» и обычные районы, граничащие с городом своего региона; src/mo/agglo.py)
три меры, знак везде «больше = больше похож на свой город»:
  closeness         = −ln(d_city / d_peer): профиль трат ближе к городу, чем к обычным районам своего региона (P1);
  comovement_excess = corr(район, город) − медиана corr(район, обычные районы региона) по помесячным изменениям
                      относительного уровня трат: синхронность именно с городом, сверх регионального фона;
  wage_resid        = остаток ln(траты район/город) после регрессии на ln(зарплата район/город) по контролю:
                      жители тратят больше, чем объясняет зарплата по месту работы.
Индекс спутника — среднее z-оценок мер. Проверка: AUC «бублики против обычных районов» (бутстрап-ДИ).
«Скрытые спутники» — обычные районы с индексом не ниже медианы «бубликов». Их внешняя проверка, которая в индекс
не входит: расстояние по дороге до города, рост населения по Росстату (на 1 января, только МО без смены границ),
доля горожан. Масштаб: «функциональный город» = город + его «бублик» + скрытые спутники.
Расширение проверки: пары «бублик — город» вне полной панели, где у обоих МО не меньше 12 месяцев данных
(профиль и синхронность по доступным месяцам, стандартизация по параметрам полной панели).
Неопределенность: бутстрап месяцев — интервалы для числа скрытых спутников и численности функциональных городов,
вероятность «скрытый спутник» для каждого МО.
Выход: outputs/stage3/functional_cities.json, outputs/stage3/satellites.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import agglo, data, features, links, rosstat, stats  # noqa: E402
from mo.config import load, path  # noqa: E402


def auc_ci(score: pd.Series, y: pd.Series, n: int, seed: int) -> tuple[float, list[float]]:
    ok = score.notna()
    s, t = score[ok].to_numpy(), y[ok].to_numpy()
    rng = np.random.default_rng(seed)
    pos, neg = np.where(t == 1)[0], np.where(t == 0)[0]
    boot = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                          np.r_[s[rng.choice(pos, len(pos))], s[rng.choice(neg, len(neg))]]) for _ in range(n)]
    return round(float(roc_auc_score(t, s)), 3), [round(float(np.quantile(boot, q)), 3) for q in (0.025, 0.975)]


def wide_any(c: pd.DataFrame, ids: list[int], cfg: dict) -> pd.DataFrame:
    """Траты по категориям для любых МО (пропуски месяцев допустимы), как panel.wide()."""
    pc = cfg["panel"]
    w = (c[c["territory_id"].isin(ids)]
         .pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="first"))
    w = w.reindex(columns=pc["categories"] + [pc["total_category"]]).astype(float).dropna()
    w.insert(len(pc["categories"]), pc["other_category"], w[pc["total_category"]] - w[pc["categories"]].sum(axis=1))
    return w.sort_index()


def main() -> None:
    cfg = load()
    fc, seed = cfg["functional"], cfg["seed"]
    nb, npm = cfg["probes"]["bootstrap"], cfg["probes"]["permutations"]
    proc, out = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    rp = pd.read_parquet(proc / "ring_pairs.parquet")
    u = agglo.units(mo, rp, pd.read_parquet(proc / "edges_contiguity.parquet"))
    pool = agglo.peer_pool(mo, rp)
    region = mo["region_code"]
    res = {"definition": __doc__.split("Выход")[0].strip()}

    # --- Три меры на полной панели ---
    prof = pd.read_parquet(proc / "profile.parquet")
    X = features.standardize(prof)
    pc = agglo.profile_closeness(X, region, u, pool).set_index("unit")
    lvl = pd.read_parquet(proc / "features_monthly.parquet")["rel_level"].unstack("date").loc[ids]
    dl = lvl.diff(axis=1).iloc[:, 1:]
    cm = agglo.comovement(dl, region, u, pool, fc["comovement_min_obs"]).set_index("unit")
    w = pd.read_parquet(proc / "panel_wide.parquet")
    total = cfg["panel"]["total_category"]
    year = cfg["features"]["profile_year"]
    spend = w[w.index.get_level_values("date").str.startswith(year)][total].groupby(level="territory_id").mean()
    wr = agglo.wage_residual(spend, mo["wage"], u).set_index("unit")
    sig = u.set_index("unit")[["group", "partner", "road_km"]].copy()
    sig["closeness"] = -np.log(pc["ratio"])
    sig["comovement_excess"] = cm["excess"]
    sig["corr_city"] = cm["corr_city"]
    sig["wage_resid"] = wr["resid"]
    sig["index"] = agglo.satellite_index(sig, fc["signals"], fc["min_signals"])
    y = (sig["group"] == "ring").astype(int)

    res["n_pairs"] = {"ring": int((y == 1).sum()), "control": int((y == 0).sum()),
                      "index_defined": int(sig["index"].notna().sum())}
    res["wage_fit_on_control"] = {"a": round(float(wr["fit_a"].iloc[0]), 4), "b": round(float(wr["fit_b"].iloc[0]), 4)}
    auc = {}
    for col in fc["signals"] + ["index"]:
        a, ci = auc_ci(sig[col], y, nb, seed)
        auc[col] = {"auc": a, "ci95": ci}
    res["auc_ring_vs_control"] = auc
    res["signal_correlations"] = sig[fc["signals"]].corr().round(3).to_dict()
    res["comovement_excess_mean"] = {g: round(float(v.mean()), 3) for g, v in sig.groupby("group")["comovement_excess"]}
    res["comovement_excess_perm"] = stats.perm_test_diff(sig.loc[y == 1, "comovement_excess"].dropna(),
                                                         sig.loc[y == 0, "comovement_excess"].dropna(), n=npm, seed=seed)

    # --- Скрытые спутники ---
    q = fc["hidden_threshold_ring_quantile"]
    thr = float(sig.loc[y == 1, "index"].quantile(q))
    ctrl = sig[y == 0].copy()
    ctrl["hidden"] = ctrl["index"] >= thr
    b = data.borders()
    g0, g1 = fc["growth_years"]
    p0, p1 = rosstat.population_jan1(b, g0), rosstat.population_jan1(b, g1)
    growth = (p1 / p0 - 1).dropna()
    sig["pop_growth"] = growth.reindex(sig.index)
    ctrl["pop_growth"] = sig["pop_growth"]
    ctrl["urban_share"] = mo["urban_share"].reindex(ctrl.index)
    ctrl["population"] = mo["population"].reindex(ctrl.index)
    hid, rest = ctrl[ctrl["hidden"]], ctrl[~ctrl["hidden"] & ctrl["index"].notna()]

    def cmp(col: str) -> dict:
        a_, b_ = hid[col].dropna(), rest[col].dropna()
        return {"hidden_median": round(float(a_.median()), 4), "other_control_median": round(float(b_.median()), 4),
                "ring_median": round(float(sig.loc[y == 1, col].dropna().median()), 4) if col in sig else None,
                "n_hidden": int(len(a_)), "n_other": int(len(b_)),
                "perm_median_diff": stats.perm_test_diff(a_, b_, stat=np.median, n=npm, seed=seed)}

    sens = {}
    for qq in fc["sensitivity_quantiles"]:
        t = float(sig.loc[y == 1, "index"].quantile(qq))
        sens[str(qq)] = {"threshold": round(t, 3), "n_hidden": int((ctrl["index"] >= t).sum())}
    auc_ok = auc["index"]["auc"] >= fc["auc_candidates_only_below"]
    two = agglo.satellite_index(sig, ["closeness", "comovement_excess"], 2)
    a2, ci2 = auc_ci(two, y, nb, seed)
    t2 = float(two[y == 1].quantile(q))
    sens["without_wage_signal"] = {"auc": a2, "ci95": ci2, "n_hidden": int((two[y == 0] >= t2).sum()),
                                   "overlap_with_main": int(((two[y == 0] >= t2) & (ctrl["index"] >= thr)).sum())}
    res["hidden_satellites"] = {
        "threshold_index": round(thr, 3), "rule": f"индекс не ниже квантиля {q} индекса «бубликов»",
        "status": "спутники" if auc_ok else "только кандидаты (AUC ниже порога)",
        "n": int(len(hid)), "share_of_control": round(float(len(hid) / ctrl["index"].notna().sum()), 3),
        "n_center_within_5km": int((hid["road_km"] < 5).sum()),
        "center_within_5km": [mo.loc[t, "name"] for t in hid.index[hid["road_km"] < 5]],
        "population_mln": round(float(hid["population"].sum()) / 1e6, 3),
        "urban_share_weighted": round(float((hid["population"] * hid["urban_share"]).sum()
                                            / hid.loc[hid["urban_share"].notna(), "population"].sum()), 3),
        "validation_not_in_index": {"road_km": cmp("road_km"), "pop_growth": cmp("pop_growth"),
                                    "urban_share": cmp("urban_share")},
        "threshold_sensitivity": sens,
    }
    res["pop_growth"] = {
        "years": [g0, g1], "n_mo_with_growth": int(growth.reindex(ids).notna().sum()),
        "median_all_panel_rural": round(float(growth.reindex(list(pool)).dropna().median()), 4),
        "median_ring": round(float(sig.loc[y == 1, "pop_growth"].dropna().median()), 4),
        "median_control": round(float(sig.loc[y == 0, "pop_growth"].dropna().median()), 4),
        "perm_ring_vs_control_median": stats.perm_test_diff(sig.loc[y == 1, "pop_growth"].dropna(),
                                                            sig.loc[y == 0, "pop_growth"].dropna(),
                                                            stat=np.median, n=npm, seed=seed),
    }

    # --- Масштаб: функциональные города ---
    rp_all = links.ring_pairs(b[b["year_to"] == 9999])
    pop_all = rosstat.population(b, cfg["rosstat"]["year"])[0]["population"]
    latest = data.borders_latest(b)
    # Только пары, где траты есть у обоих (полная панель + расширенная проверка ниже): без данных поведение не проверить
    c = data.consumption()
    nm = c[c["category"] == total].groupby("territory_id")["date"].nunique()
    ext = rp_all[~rp_all["ring"].isin(rp["ring"])].copy()
    ext = ext[(ext["ring"].map(nm).fillna(0) >= fc["ext_min_months"]) & (ext["city"].map(nm).fillna(0) >= fc["ext_min_months"])]
    rings_data = pd.concat([rp[["ring", "city"]], ext[["ring", "city"]]])
    attach = pd.concat([pd.DataFrame({"city": rings_data["city"], "unit": rings_data["ring"], "kind": "ring"}),
                        pd.DataFrame({"city": hid["partner"].astype(int), "unit": hid.index.astype(int), "kind": "hidden"})],
                       ignore_index=True)
    attach["pop"] = pop_all.reindex(attach["unit"]).to_numpy()
    fcity = (attach.pivot_table(index="city", columns="kind", values="pop", aggfunc="sum")
             .reindex(columns=["ring", "hidden"]).fillna(0).rename(columns={"ring": "ring_pop", "hidden": "hidden_pop"}))
    fcity["n_hidden"] = attach[attach["kind"] == "hidden"].groupby("city").size().reindex(fcity.index).fillna(0).astype(int)
    fcity["city_pop"] = pop_all.reindex(fcity.index)
    fcity["functional_pop"] = fcity["city_pop"] + fcity["ring_pop"] + fcity["hidden_pop"]
    fcity["ratio"] = fcity["functional_pop"] / fcity["city_pop"]
    fcity["name"] = latest["municipal_district_name"].reindex(fcity.index)
    fcity["center"] = latest["municipal_district_center"].reindex(fcity.index).str.replace(r"^г\.?\s+", "", regex=True)
    fcity["n_ring"] = attach[attach["kind"] == "ring"].groupby("city").size().reindex(fcity.index).fillna(0).astype(int)
    fcity["added_pop"] = fcity["ring_pop"] + fcity["hidden_pop"]
    fcity = fcity.dropna(subset=["city_pop"]).sort_values("functional_pop", ascending=False)
    res["functional_cities"] = {
        "rule": "город + его «бублик» (если у обоих есть траты) + скрытые спутники; население Росстата на 1.01.2023",
        "n_cities": int(len(fcity)),
        "official_city_pop_mln": round(float(fcity["city_pop"].sum()) / 1e6, 3),
        "ring_pop_mln": round(float(fcity["ring_pop"].sum()) / 1e6, 3),
        "hidden_pop_mln": round(float(fcity["hidden_pop"].sum()) / 1e6, 3),
        "functional_pop_mln": round(float(fcity["functional_pop"].sum()) / 1e6, 3),
        "median_ratio": round(float(fcity["ratio"].median()), 3),
        "top_by_ratio_city_over_100k": [
            {"city": r.name, "city_pop": int(r.city_pop), "functional_pop": int(r.functional_pop), "ratio": round(r.ratio, 2),
             "n_hidden": int(r.n_hidden)}
            for r in fcity[fcity["city_pop"] >= 1e5].sort_values("ratio", ascending=False).head(15).itertuples()],
        "largest_added_city_over_250k": [
            {"city": r.name, "center": r.center, "city_pop": int(r.city_pop), "functional_pop": int(r.functional_pop),
             "ratio": round(r.ratio, 2), "n_ring": int(r.n_ring), "n_hidden": int(r.n_hidden)}
            for r in fcity[fcity["city_pop"] >= 2.5e5].sort_values("added_pop", ascending=False).head(12).itertuples()],
        "largest_with_hidden": [
            {"city": r.name, "city_pop": int(r.city_pop), "functional_pop": int(r.functional_pop), "ratio": round(r.ratio, 2),
             "n_hidden": int(r.n_hidden)}
            for r in fcity[fcity["n_hidden"] > 0].sort_values("city_pop", ascending=False).head(15).itertuples()],
    }

    # --- Неопределенность: бутстрап месяцев (профиль 2024, изменения трат, траты для зарплатной меры) ---
    B = fc["bootstrap_months"]
    if B:
        rng = np.random.default_rng(seed)
        dates = w.index.get_level_values("date")
        m24 = sorted(d_ for d_ in dates.unique() if d_.startswith(year))
        cats = features.share_cols()
        sh3 = np.stack([features.shares(w.xs(m, level="date").loc[ids]).to_numpy() for m in m24])
        lg = np.log(w[total])
        rel = (lg - lg.groupby(level="date").transform("median"))
        rel2 = np.stack([rel.xs(m, level="date").loc[ids].to_numpy() for m in m24])
        tot2 = np.stack([w.xs(m, level="date").loc[ids, total].to_numpy() for m in m24])
        ring_pop = float(pop_all.reindex(rings_data["ring"]).sum())
        cities_ring = set(rings_data["city"])
        hid_cnt = pd.Series(0.0, index=ctrl.index)
        draws = []
        for _ in range(B):
            mi = rng.integers(0, len(m24), len(m24))
            msh = pd.DataFrame(sh3[mi].mean(0), index=ids, columns=cats)
            prof_b = pd.concat([features.clr(msh), pd.Series(rel2[mi].mean(0), index=ids, name="rel_level")], axis=1)[prof.columns]
            Xb = features.standardize(prof_b)
            dl_b = dl.iloc[:, rng.integers(0, dl.shape[1], dl.shape[1])]
            spend_b = pd.Series(tot2[mi].mean(0), index=ids)
            pcb = agglo.profile_closeness(Xb, region, u, pool).set_index("unit")
            cmb = agglo.comovement(dl_b, region, u, pool, fc["comovement_min_obs"]).set_index("unit")
            wrb = agglo.wage_residual(spend_b, mo["wage"], u).set_index("unit")
            sb = pd.DataFrame({"closeness": -np.log(pcb["ratio"]), "comovement_excess": cmb["excess"],
                               "wage_resid": wrb["resid"]}).reindex(sig.index)
            ib = agglo.satellite_index(sb, fc["signals"], fc["min_signals"])
            thr_b = float(ib[y == 1].quantile(q))
            hb = ib[y == 0] >= thr_b
            hid_cnt += hb.reindex(hid_cnt.index).fillna(False).astype(float)
            hs_ids = hb.index[hb]
            cities = cities_ring | set(sig.loc[hs_ids, "partner"].astype(int))
            off = float(pop_all.reindex(list(cities)).sum())
            hp = float(pop_all.reindex(hs_ids).sum())
            draws.append({"n_hidden": int(hb.sum()), "hidden_pop": hp, "functional_pop": off + ring_pop + hp,
                          "ratio": (off + ring_pop + hp) / off})
        dr = pd.DataFrame(draws)
        ci = lambda c, k=1.0: [round(float(dr[c].quantile(0.025)) / k, 3), round(float(dr[c].quantile(0.975)) / k, 3)]  # noqa: E731
        p_hidden = hid_cnt / B
        res["uncertainty"] = {
            "method": f"бутстрап месяцев, {B} повторов: профиль 2024, помесячные изменения трат, траты для зарплатной меры",
            "n_hidden_ci95": ci("n_hidden"), "hidden_pop_mln_ci95": ci("hidden_pop", 1e6),
            "functional_pop_mln_ci95": ci("functional_pop", 1e6), "ratio_ci95": ci("ratio"),
            "hidden_point_stable_share": round(float((p_hidden[hid.index] >= 0.5).mean()), 3),
            "n_p_hidden_ge_0_9": int((p_hidden >= 0.9).sum()),
        }
        sig["p_hidden"] = p_hidden.reindex(sig.index)

    # --- Расширение проверки: пары вне полной панели ---
    # МО вне полной панели (у части пар город в панели есть: для него берем признаки панели)
    ext_ids = sorted((set(ext["ring"]) | set(ext["city"])) - set(ids))
    we = wide_any(c, ext_ids, cfg)
    lg_panel = np.log(w[total]).groupby(level="date").median()
    sh_e = features.shares(we)
    lvl_e = (np.log(we[total]) - lg_panel.reindex(we.index.get_level_values("date")).to_numpy()).rename("rel_level")
    dates = we.index.get_level_values("date")
    rows = []
    for t in ext_ids:
        m = we.index.get_level_values("territory_id") == t
        m24 = m & dates.str.startswith(year)
        sel = m24 if m24.sum() >= fc["ext_profile_min_months"] else m
        rows.append(pd.concat([features.clr(sh_e[sel].mean().to_frame().T), pd.Series({"rel_level": lvl_e[sel].mean()}).to_frame().T],
                              axis=1).assign(territory_id=t))
    prof_e = pd.concat(rows).set_index("territory_id")[prof.columns]
    Xall = pd.concat([X, (prof_e - prof.mean()) / prof.std(ddof=0)])
    reg_all = pd.concat([region, latest["region_code"].reindex(ext_ids)])
    ue = pd.DataFrame({"group": "ring", "unit": ext["ring"].to_numpy(), "partner": ext["city"].to_numpy(), "road_km": 0.0})
    pce = agglo.profile_closeness(Xall, reg_all, ue, pool)
    lvl_all = pd.concat([lvl, lvl_e.unstack("date").reindex(columns=lvl.columns)])
    cme = agglo.comovement(lvl_all.diff(axis=1).iloc[:, 1:], reg_all, ue, pool, fc["comovement_min_obs"])
    ring_p = sig[y == 1]
    ctl = sig[y == 0]
    closer_ctl = (pc.loc[pc["group"] == "control", "closer"]).astype(float)
    closer_ring_all = pd.concat([pc.loc[pc["group"] == "ring", "closer"], pce["closer"]]).astype(float)
    corr_ring_all = pd.concat([ring_p["corr_city"], cme["corr_city"]]).dropna()
    res["extended_check"] = {
        "rule": f"пары «бублик — город» вне полной панели, у обоих МО ≥ {fc['ext_min_months']} мес.",
        "n_pairs_ext": int(len(ext)), "n_profile": int(len(pce)), "n_comovement": int(cme["corr_city"].notna().sum()),
        "share_closer_ext": round(float(pce["closer"].mean()), 3) if len(pce) else None,
        "mean_corr_city_ext": round(float(cme["corr_city"].mean()), 3),
        "combined": {
            "n_profile": int(len(closer_ring_all)), "share_closer": round(float(closer_ring_all.mean()), 3),
            "share_closer_ci95": [round(v, 3) for v in stats.bootstrap_ci(closer_ring_all, n=nb, seed=seed)],
            "control_share_closer": round(float(closer_ctl.mean()), 3),
            "perm_closer_vs_control": stats.perm_test_diff(closer_ring_all, closer_ctl, n=npm, seed=seed),
            "n_comovement": int(len(corr_ring_all)), "mean_corr_city": round(float(corr_ring_all.mean()), 3),
            "control_mean_corr_city": round(float(ctl["corr_city"].mean()), 3),
            "perm_corr_vs_control": stats.perm_test_diff(corr_ring_all, ctl["corr_city"].dropna(), n=npm, seed=seed),
        },
        "pairs": [{"ring": latest.loc[r, "municipal_district_name"], "city": latest.loc[ci, "municipal_district_name"],
                   "months_ring": int(nm.get(r, 0)), "months_city": int(nm.get(ci, 0))}
                  for r, ci in zip(ext["ring"], ext["city"])],
    }

    # --- Выгрузка ---
    names = mo["name"]
    hid_out = hid.assign(name=names.reindex(hid.index), city=names.reindex(hid["partner"].astype(int)).to_numpy(),
                         region=mo["region"].reindex(hid.index)).sort_values("index", ascending=False)
    res["hidden_satellites"]["top"] = [
        {"name": r.name, "city": r.city, "region": r.region, "index": round(r.index, 2), "road_km": round(r.road_km, 1),
         "population": int(r.population) if np.isfinite(r.population) else None,
         "pop_growth": round(r.pop_growth, 4) if np.isfinite(r.pop_growth) else None}
        for r in hid_out.head(20).itertuples()]
    sig_out = sig.assign(name=names.reindex(sig.index), city=names.reindex(sig["partner"].astype(int)).to_numpy(),
                         hidden=sig.index.isin(hid.index))
    sig_out.index.name = "territory_id"
    sig_out.round(4).to_csv(out / "satellites.csv")
    (out / "functional_cities.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("definition",)}, ensure_ascii=False, indent=1, default=float)[:7000])


if __name__ == "__main__":
    main()
