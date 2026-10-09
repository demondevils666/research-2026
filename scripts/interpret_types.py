"""Этап 3, шаг 1: интерпретация шести типов.

Для итоговой статической типологии (KEFRiN на физической сети) считает:
  размеры и население типов; медианные траты на жителя и доли категорий (2024); прирост доли маркетплейсов 2024/2023;
  профиль по правилу Миркина (отклонение среднего по типу от среднего по стране, %);
  внешнюю проверку признаками, которые НЕ участвовали в кластеризации: Росстат (население, доля горожан, зарплата,
  структура занятости), индекс доступности рынков, индекс мобильности (только Северо-Запад) — η² и Краскел–Уоллис;
  состав по типам МО и регионам; самые крупные и самые типичные МО;
  интервальные правила для каждого типа (этап 3b, E3): «доля X ≥ a и траты ≤ b» с точностью, полнотой и F1.
Выход: outputs/stage3/types.json, outputs/stage3/mo_types.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, interpret  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    proc, out = path("processed"), path("outputs") / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    names = interpret.type_names(lab)
    code = lab.map(names["code"])

    w = pd.read_parquet(proc / "panel_wide.parquet")
    total = cfg["panel"]["total_category"]
    by_year = {y: w[w.index.get_level_values("date").str.startswith(y)].groupby(level="territory_id").mean().loc[ids]
               for y in ("2023", "2024")}
    wy = by_year["2024"]
    sh = features.shares(wy) * 100
    mp_gr = sh["Маркетплейсы"] - features.shares(by_year["2023"])["Маркетплейсы"] * 100
    spend = wy[features.share_cols() + [total]]

    ctx_cols = ["population", "urban_share", "wage", "emp_agri", "emp_mining", "emp_industry", "emp_market_services",
                "emp_public", "market_access"]
    ctx = mo.loc[ids, ctx_cols].copy()
    ctx["log_population"] = np.log(ctx["population"])
    ctx["log_wage"] = np.log(ctx["wage"])
    mob = pd.read_parquet(path("raw") / "sberindex_api" / "indeks-mobilnosti.parquet")
    mob = mob[mob["period"].str.startswith("2024")].groupby("ref_area")["value"].mean()
    dup = mo["name"].value_counts()
    uniq = mo.loc[ids, "name"].map(lambda n: dup[n] == 1)
    ctx["mobility_km"] = mo.loc[ids, "name"].map(mob).where(uniq)

    res = {"K": int(lab.nunique()), "types": {}}
    mk_spend = interpret.mirkin(spend, code)
    mk_share = interpret.mirkin(sh, code)
    mk_ctx = interpret.mirkin(ctx[ctx_cols], code)
    typ = interpret.typical(features.standardize(pd.read_parquet(proc / "profile.parquet")).loc[ids], code)
    for c in sorted(code.unique()):
        idx = code.index[code == c]
        m = mo.loc[idx]
        row = names[names["code"] == c].iloc[0]
        res["types"][c] = {
            "name": row["name"], "label": int(names.index[names["code"] == c][0]), "note": row["note"],
            "n_mo": int(len(idx)), "population_mln": round(float(m["population"].sum()) / 1e6, 4),   # 4 знака: без двойного округления в тексте
            "spend_rub_median": int(spend.loc[idx, total].median()),
            "shares_median_pct": {k: round(float(v), 1) for k, v in sh.loc[idx].median().items()},
            "marketplace_share_growth_pp_median": round(float(mp_gr.loc[idx].median()), 2),
            "context_median": {k: (round(float(v), 3) if k not in ("population", "wage") else int(v))
                               for k, v in m[ctx_cols].median().items()},
            "mobility_km_median": (round(float(ctx.loc[idx, "mobility_km"].median()), 2)
                                   if ctx.loc[idx, "mobility_km"].notna().sum() >= 5 else None),
            "mirkin_spend_per_capita_pct": mk_spend.loc[c].to_dict(),
            "mirkin_shares_pct": mk_share.loc[c].to_dict(),
            "mirkin_context_pct": mk_ctx.loc[c].to_dict(),
            "mo_kinds": m["type"].value_counts().to_dict(),
            "n_regions": int(m["region"].nunique()),
            "top_regions": m["region"].value_counts().head(5).to_dict(),
            "largest": m.sort_values("population", ascending=False)["name"].head(6).tolist(),
            "most_typical": mo.loc[typ[c], "name"].tolist(),
        }
    # E3: интервальные правила (доли категорий в %, траты на жителя в ₽, 2024)
    rc = cfg["rules"]
    Xr = pd.concat([sh, spend[[total]]], axis=1)
    steps = {c: rc["round_steps"].get(c, rc["round_steps"]["default"]) for c in Xr.columns}
    rules = interpret.interval_rules(Xr, code, rc["quantiles"], steps, rc["max_terms"], rc["beam"], rc["min_gain"],
                                     rc["folds"], cfg["seed"])
    for c, r in rules.items():
        res["types"][c]["rule"] = r
    res["rules_summary"] = {"f1_mean": round(float(np.mean([r["f1"] for r in rules.values()])), 3),
                            "f1_cv_mean": round(float(np.mean([r["f1_cv_mean"] for r in rules.values()])), 3),
                            "note": "правило — конъюнкция до 3 интервальных условий; F1 на отложенных МО — 5 фолдов"}
    ext = {k: interpret.eta2(ctx[k], code) for k in
           ["log_population", "urban_share", "log_wage", "emp_agri", "emp_mining", "emp_industry", "emp_market_services",
            "emp_public", "market_access", "mobility_km"]}
    # База сравнения: столько же групп только по уровню трат (квантили относительного уровня за год профиля). Уровень сам
    # входит в признаки и связан с зарплатой, поэтому тип «добавляет» к внешней проверке только то, что выше этой базы.
    lvl = features.rel_level(w)
    lvl = lvl[lvl.index.get_level_values("date").str.startswith(cfg["features"]["profile_year"])].groupby(level="territory_id").mean().loc[ids]
    k_types = int(code.nunique())
    level_groups = pd.qcut(lvl, k_types, labels=False)
    fed = mo.loc[ids, "type"].str.startswith("внутригородская")
    for k in ext:
        ext[k]["eta2_level_groups"] = interpret.eta2(ctx[k], level_groups)["eta2"]
        # 95%-интервал разницы «тип минус база» (бутстрап МО): «тип сильнее базы», только если интервал выше нуля
        ext[k]["diff_ci95"] = [round(v, 4) for v in interpret.eta2_diff_ci(ctx[k], code, level_groups,
                                                                           rc["ext_bootstrap"], cfg["seed"])]
        ext[k]["eta2_no_fed_cities"] = interpret.eta2(ctx[k][~fed], code[~fed])["eta2"]
        ext[k]["eta2_level_groups_no_fed_cities"] = interpret.eta2(ctx[k][~fed], pd.qcut(lvl[~fed], k_types, labels=False))["eta2"]
    res["external_validation"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                  for k, v in ext.items()}
    res["external_validation_note"] = ("Ни один из этих показателей не участвовал в кластеризации (признаки — только траты "
                                       "СберИндекса). η² — доля различий между МО, которую объясняет тип. База сравнения — "
                                       "столько же групп только по уровню трат (eta2_level_groups): тип ценен там, где он выше базы.")
    for c in res["types"]:
        res["types"][c]["n_fed_city_districts"] = int(((code == c) & fed).sum())
    # Сравнение с административным типом МО (насколько типология повторяет административную сетку)
    from sklearn.metrics import adjusted_rand_score
    res["ari_with_admin_kind"] = round(adjusted_rand_score(mo.loc[ids, "type"], code), 3)
    res["ari_with_region"] = round(adjusted_rand_score(mo.loc[ids, "region_code"], code), 3)
    (out / "types.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    tab = pd.DataFrame({"territory_id": ids, "name": mo.loc[ids, "name"].to_numpy(), "region": mo.loc[ids, "region"].to_numpy(),
                        "kind": mo.loc[ids, "type"].to_numpy(), "type": code.to_numpy(),
                        "type_name": code.map(names.set_index("code")["name"]).to_numpy()})
    tab.to_csv(out / "mo_types.csv", index=False)
    for c, t in res["types"].items():
        print(c, t["name"], t["n_mo"], "МО,", t["population_mln"], "млн; траты", t["spend_rub_median"], "; Миркин доли:",
              {k[:6]: v for k, v in t["mirkin_shares_pct"].items()})
        print("    типичные:", t["most_typical"][:3], "| контекст Миркин:", {k: v for k, v in t["mirkin_context_pct"].items() if abs(v) > 30})
    for c, r in rules.items():
        txt = " и ".join(f"{t['feature']} {'≥' if t['op'] == '>=' else '≤'} {t['threshold']:g}" for t in r["rule"])
        print(f"{c}: {txt} | точность {r['precision']}, полнота {r['recall']}, F1 {r['f1']} (CV {r['f1_cv_mean']}), доля {r['prevalence']}")
    print(json.dumps(res["external_validation"], ensure_ascii=False))
    print("ARI с типом МО:", res["ari_with_admin_kind"], "с регионом:", res["ari_with_region"])


if __name__ == "__main__":
    main()
