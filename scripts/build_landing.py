"""Лендинг: один HTML-файл без внешних запросов (данные, геометрия МО и шрифты встроены) -> site/index.html.

Собирает из готовых выходов пайплайна: типы (статика, по месяцам, по весу сети α), «бублики» и скрытые спутники,
двойники, Росстат (доля горожан, зарплата, население), таблицы для графиков (три меры агломераций, развертка α,
методы и ICVI, динамика). Геометрия МО: проекция Альберса для России, упрощение, целые координаты.
Шаблон страницы: <out_dir>/template.html (разметка, стили, скрипт), out_dir задан в configs/base.yaml (site; в репозитории
подачи — docs для GitHub Pages). Отчет копируется рядом (<out_dir>/report.pdf).
Запуск: python scripts/build_landing.py
"""

import base64
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import data, features, interpret, links, methods, numfmt  # noqa: E402
from mo.config import load, path  # noqa: E402
from mo.names import short  # noqa: E402

ALBERS = "+proj=aea +lat_0=0 +lon_0=100 +lat_1=50 +lat_2=70 +ellps=WGS84 +units=m"


def j(p: Path):
    return json.loads(p.read_text())


def geometry(cfg: dict, regions: pd.Series, focus: list[int] | None = None) -> dict:
    """Пути SVG всех действующих МО и рамки предустановленных видов карты (по точкам МО внутри полигона).
    focus — МО витрины (город и его район): квадратная рамка вокруг них и список МО, которые в нее попадают."""
    lc = cfg["landing"]
    g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last")
    g = g[g["year_to"] == 9999].set_index("territory_id")
    pt = g.geometry.representative_point()
    lon, lat = pt.x % 360, pt.y
    g = g.to_crs(ALBERS)
    # Упрощение покрытия (Висвалингам — Уайатт): общие границы соседей упрощаются одинаково, без щелей между МО
    g["geometry"] = shapely.coverage_simplify(g.geometry.to_numpy(), lc["simplify_m"])
    x0, y0, x1, y1 = g.total_bounds
    s = lc["map_width"] / (x1 - x0)
    W, H = lc["map_width"], int(round((y1 - y0) * s))
    views = []
    for v in lc["views"]:
        if "region" in v:
            sel = regions.reindex(g.index) == v["region"]
        elif "lon" in v:
            sel = lon.between(*v["lon"]) & lat.between(*v["lat"])
        else:
            views.append({"name": v["name"], "box": [0, 0, W, H]})
            continue
        bx0, by0, bx1, by1 = g[sel.to_numpy()].total_bounds
        pad = lc["view_pad"] * max(bx1 - bx0, by1 - by0)
        views.append({"name": v["name"], "box": [int((bx0 - pad - x0) * s), int((y1 - by1 - pad) * s),
                                                 int((bx1 - bx0 + 2 * pad) * s), int((by1 - by0 + 2 * pad) * s)]})
    fbox, fids, fpts = None, [], {}
    if focus:
        bx0, by0, bx1, by1 = g.loc[focus].total_bounds
        cx, cy, half = (bx0 + bx1) / 2, (by0 + by1) / 2, max(bx1 - bx0, by1 - by0) * 0.75
        fids = [int(t) for t in g.cx[cx - half:cx + half, cy - half:cy + half].index]
        fbox = [int((cx - half - x0) * s), int((y1 - cy - half) * s), int(2 * half * s), int(2 * half * s)]
        # точки подписей: внутри полигона (у района — с отступом от границы, чтобы подпись не легла на город)
        for t in focus:
            inner = g.loc[t].geometry.buffer(-0.08 * half)
            pt_ = (inner if not inner.is_empty else g.loc[t].geometry).representative_point()
            fpts[int(t)] = [int((pt_.x - x0) * s), int((y1 - pt_.y) * s)]
    paths = {}
    for tid, geom in g.geometry.items():
        if geom is None or geom.is_empty:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        parts = []
        for p in polys:
            for ring in [p.exterior] + list(p.interiors):
                xy = np.asarray(ring.coords)
                px = np.round((xy[:, 0] - x0) * s).astype(int)
                py = np.round((y1 - xy[:, 1]) * s).astype(int)
                keep = np.r_[True, (np.diff(px) != 0) | (np.diff(py) != 0)]
                px, py = px[keep], py[keep]
                if len(px) < 3:
                    continue
                d = [f"M{px[0]} {py[0]}"] + [f"l{dx} {dy}" for dx, dy in zip(np.diff(px), np.diff(py))]
                parts.append("".join(d) + "z")
        if parts:
            paths[int(tid)] = "".join(parts)
    # узлы сети для слоя ребер: точка внутри полигона, в тех же пикселях карты, что и пути
    nodes = {int(t): [int((q.x - x0) * s), int((y1 - q.y) * s)] for t, q in g.geometry.representative_point().items()
             if q is not None and not q.is_empty}
    return {"w": W, "h": H, "paths": paths, "views": views, "focus_box": fbox, "focus_ids": fids, "focus_pts": fpts,
            "nodes": nodes}


SBER_EXT = {"market_access", "mobility_km"}   # индексы СберИндекса во внешней проверке; остальные показатели — Росстат


def main() -> None:
    cfg = load()
    proc, o2, o3 = path("processed"), path("outputs") / "stage2", path("outputs") / "stage3"
    site = ROOT / cfg["landing"]["out_dir"]     # шаблон, шрифты и результат лежат в одной папке
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    st = pd.read_csv(o2 / "labels_static.csv", index_col="territory_id").loc[ids, "type"]
    names = interpret.type_names(st)
    code_of = names["code"].to_dict()
    tcfg = yaml.safe_load((ROOT / "configs" / "types.yaml").read_text())["types"]
    dark = cfg["landing"]["dark_colors"]
    digit = {t["code"]: str(i + 1) for i, t in enumerate(sorted(tcfg, key=lambda t: t["code"]))}
    tj = j(o3 / "types.json")
    prj = j(o3 / "patterns_rules.json") if (o3 / "patterns_rules.json").exists() else None
    types = []
    for t in sorted(tcfg, key=lambda t: t["code"]):
        r = tj["types"][t["code"]]
        types.append({"code": t["code"], "name": t["name"], "short": t["short"], "color": t["color"],
                      "dark": dark[t["code"]], "note": t["note"], "n": r["n_mo"], "pop": r["population_mln"],
                      "spend": r["spend_rub_median"], "shares": r["shares_median_pct"],
                      "mirkin": r["mirkin_shares_pct"], "mirkin_spend": r["mirkin_spend_per_capita_pct"]["Все категории"],
                      "rule": r["rule"], "largest": [short(x) for x in r["largest"][:4]],
                      "stab": prj["rule_stability"][t["code"]]["extent_jaccard_median"] if prj else None,
                      "own": prj["ordinal_patterns"]["closest_is_own_share"][t["code"]] if prj else None})
    # Метки: статика, по месяцам, по весу сети α (выравниваются к итоговым типам)
    LM = pd.read_csv(o2 / "labels_monthly.csv", index_col="territory_id").loc[ids]
    months = list(LM.columns)
    m_str = ["".join(digit[code_of[v]] for v in LM.loc[t]) for t in ids]
    # Известные шоки 2024 года (configs: report.shocks_2024): годовой тип против самого частого помесячного
    shock_notes = []
    for tid in ids:
        name = mo.loc[tid, "name"]
        if name in cfg["report"].get("shocks_2024", {}):
            mc = LM.loc[tid].map(code_of).value_counts()
            if mc.index[0] != code_of[st[tid]] and mc.iloc[0] >= 0.75 * len(months):
                shock_notes.append({"n": short(name), "t": code_of[st[tid]], "m": mc.index[0], "k": int(mc.iloc[0]),
                                    "of": len(months), "why": cfg["report"]["shocks_2024"][name]})
    SW = pd.read_parquet(proc / "labels_sweep.parquet").loc[ids]
    # что сделала сеть: МО, сменившие тип между α = 0 и выбранным α (как в отчете, src/mo/methods.py)
    sw_col = {round(float(c[1:]), 4): c for c in SW.columns}
    ref = st.to_numpy()
    a_cols = list(SW.columns)
    a_lab = np.column_stack([methods.align_labels(ref, SW[c].to_numpy()) for c in a_cols])
    a_str = ["".join(digit[code_of[v]] if v in code_of else "0" for v in row) for row in a_lab]
    sweep = j(o2 / "kefrin_sweep.json")
    # Пригороды
    b = data.borders()
    rp_all = links.ring_pairs(b[b["year_to"] == 9999])
    ring_of = {int(r): int(c) for r, c in zip(rp_all["ring"], rp_all["city"])}
    sat = pd.read_csv(o3 / "satellites.csv", index_col="territory_id")
    hid = sat[sat["hidden"]]
    hidden = {int(t): [int(r.partner), round(float(r.p_hidden), 2) if "p_hidden" in sat and pd.notna(r.p_hidden) else None]
              for t, r in hid.iterrows()}
    # Близость профиля трат района у города к своему городу: d_peer / d_city = exp(closeness) (scripts/functional_cities.py)
    close = {int(t): [int(r.partner), round(float(np.exp(r.closeness)), 2)] for t, r in sat.iterrows() if pd.notna(r.closeness)}
    # МО с неполными рядами (scripts/assign_partial.py): тип по ближайшему центру, «нетипичный профиль» или мало данных
    ptj = j(o3 / "partial_types.json") if (o3 / "partial_types.json").exists() else None
    ptc = pd.read_csv(o3 / "partial_types.csv", index_col="territory_id") if ptj else None
    tw = pd.read_csv(o3 / "twins.csv", index_col="territory_id")
    twins = {int(t): [int(x) for x in str(r.twins).split(";")] for t, r in tw.iterrows()}
    # Профиль трат МО за год типологии: доли шести категорий (доли средних за год, как в interpret_types) и траты
    w = pd.read_parquet(proc / "panel_wide.parquet")
    year = cfg["features"]["profile_year"]
    wy = w[w.index.get_level_values("date").str.startswith(year)].groupby(level="territory_id").mean().loc[ids]
    sh = features.shares(wy) * 100
    spend = wy[cfg["panel"]["total_category"]]
    # отклонение МО от среднего по всем МО, % — то же правило, что у типов (src/mo/interpret.py: mirkin): траты и 6 долей
    X = pd.concat([spend, sh], axis=1)
    dev = ((X - X.mean()) / X.mean() * 100).map(lambda v: int(numfmt.fixed(v, 0)))   # половина — от нуля, как Intl
    # Витрина первого экрана (commute.showcase): город и его район в трех «линзах»
    cm = j(o3 / "commute_check.json") if (o3 / "commute_check.json").exists() else None
    p1 = j(path("outputs") / "probes" / "p1_ring_pairs.json")
    net = pd.read_parquet(proc / "net_physical.parquet")
    # число несвязанных кусков физической сети (объясняет вырождение Leiden)
    import networkx as nx
    gph = nx.Graph()
    gph.add_nodes_from(ids)
    gph.add_edges_from(zip(net["a"], net["b"]))
    n_cc = nx.number_connected_components(gph)
    netmove = methods.moved_by_network(methods.align_labels(ref, SW[sw_col[0.0]].to_numpy()),
                                       methods.align_labels(ref, SW[sw_col[round(sweep["chosen_alpha"], 4)]].to_numpy()), ids, net)
    show_ids = [cm["showcase"]["city_id"], cm["showcase"]["district_id"]] if cm else None
    # Все МО справочника (для карты): в панели или нет
    lat = data.borders_latest(b)
    geo = geometry(cfg, lat["region_name"], show_ids)
    all_ids = sorted(geo["paths"])
    in_panel = set(ids)
    regions = sorted(lat["region_name"].dropna().unique())
    reg_idx = {r: i for i, r in enumerate(regions)}
    mo_rec = {}
    for t in all_ids:
        nm = lat.loc[t, "municipal_district_name"] if t in lat.index else str(t)
        rg = lat.loc[t, "region_name"] if t in lat.index else None
        rec = {"n": short(str(nm)), "r": reg_idx.get(rg, -1)}
        if t in in_panel:
            i = ids.index(t)
            rec.update({"t": digit[code_of[st[t]]], "m": m_str[i], "a": a_str[i],
                        "u": None if pd.isna(mo.loc[t, "urban_share"]) else round(float(mo.loc[t, "urban_share"]), 3),
                        "w": None if pd.isna(mo.loc[t, "wage"]) else int(mo.loc[t, "wage"]),
                        "p": None if pd.isna(mo.loc[t, "population"]) else int(mo.loc[t, "population"]),
                        "s": [round(float(v), 1) for v in sh.loc[t]], "sp": int(round(float(spend.loc[t]))),
                        "d": [int(v) for v in dev.loc[t]]})
            if t in twins:
                # Рост трат на жителя за год, %: МО, медиана двойников, медиана своего региона; вывод — с поправкой на регион
                r = tw.loc[t]
                rec["tw"] = twins[t]
                rec["g"], rec["gt"] = round(float(r.growth_pct), 1), round(float(r.twins_growth_pct), 1)
                rec["gr"], rec["ga"] = round(float(r.region_growth_pct), 1), round(float(r.gap_adj_pp), 1)
        if t in ring_of:
            rec["ring"] = ring_of[t]
        if t in hidden:
            rec["sat"] = hidden[t]
        if t in close:
            rec["c"] = close[t]
        if ptc is not None and t in ptc.index:
            r = ptc.loc[t]
            rec["tp"] = [digit[r["type"]] if r["status"] == "type" else ("0" if r["status"] == "atypical" else None),
                         int(r[f"months_{year}"]), int(r["months_total"])]
        mo_rec[t] = rec
    # Правило ребра: типы KEFRiN на шести сетях (метки уже выровнены к итоговым типам в scripts/edge_rules.py)
    er = j(o2 / "edge_rules.json") if (o2 / "edge_rules.json").exists() else None
    if er:
        LE = pd.read_parquet(proc / "labels_edge_rules.parquet").loc[ids]
        nets = list(er["networks"])
        for t in ids:
            mo_rec[t]["e"] = "".join(digit[code_of[int(LE.loc[t, n])]] for n in nets)
    cities = set(ring_of.values()) | {v[0] for v in hidden.values()}
    for c in cities:
        if c in mo_rec:
            mo_rec[c]["city"] = 1
    ag = j(o3 / "agglomerations.json")
    fc = j(o3 / "functional_cities.json")
    cmp_ = j(o2 / "compare.json")
    ms = j(o2 / "methods_seeds.json") if (o2 / "methods_seeds.json").exists() else None
    pan = j(o2 / "icvi_panel.json") if (o2 / "icvi_panel.json").exists() else None
    sd = j(o3 / "space_dynamics.json")
    dyn = j(o2 / "dynamics.json")
    cc = j(o3 / "cash_check.json")
    mn = j(o3 / "monthly_network.json")
    twj = j(o3 / "twins.json")
    rb = j(o3 / "robustness.json")
    inv = j(path("outputs") / "inventory" / "inventory.json")
    nice = {"kefrin_phys": "KEFRiN · физическая сеть", "kefrin_sim": "KEFRiN · сеть сходства", "kmeans": "k-means",
            "gmm": "GMM", "spectral_sim": "Спектральная", "leiden_phys": "Leiden · физическая сеть",
            "leiden_sim": "Leiden · сеть сходства", "canus_phys": "CANUS · физическая сеть"}
    meth = []
    for m, r in cmp_["methods"].items():
        if "indices" not in r:
            continue
        ind = r["indices"]
        meth.append({"key": m, "name": nice[m], "SW": ind["SW"], "CHN": ind["CH"] / len(ids), "S_Dbw": ind["S_Dbw"],
                     "AVI": ind["AVI@phys"], "AVU": ind["AVU@phys"], "Q": ind["Q@phys"],
                     "rank1": r["mean_rank_contest_icvi"], "rank10": ms["mean_rank"][m] if ms else None,
                     "boot": r.get("bootstrap_ari")})
    meth.sort(key=lambda x: x["rank10"] if x["rank10"] is not None else x["rank1"])
    second = sorted((v, m) for m, v in ms["mean_rank"].items() if m != ms["reference"])[0][1] if ms else None
    show = None
    if cm:
        sc = cm["showcase"]
        cid = sc["city_id"]
        ccsv = pd.read_csv(o3 / "commute_check.csv", index_col="territory_id")
        win = [t for t in geo["focus_ids"] if t in in_panel]
        show = {**{k: sc[k] for k in ("city_id", "district_id", "district_rural_pct", "wage_pct_of_city", "spend_pct_of_city",
                                       "commute_pct", "region_median_pct", "district_type", "city_type", "year")},
                "city": short(sc["city"]), "district": short(sc["district"]).replace(" р-н", " район"),
                "box": geo["focus_box"], "ids": [int(t) for t in geo["focus_ids"]], "pts": geo["focus_pts"],
                "wage": {int(t): round(float(mo.loc[t, "wage"] / mo.loc[cid, "wage"] * 100), 1) for t in win
                         if pd.notna(mo.loc[t, "wage"])},
                "spend": {int(t): round(float(spend.loc[t] / spend.loc[cid] * 100), 1) for t in win},
                "census": {int(t): round(float(ccsv.loc[t, "share_other_settlement"] * 100), 1) for t in win
                           if t in ccsv.index and ccsv.loc[t, "region"] == sc["region"]},
                "wage_year": cfg["rosstat"]["year"]}
    lab = j(o3 / "lab_check.json") if (o3 / "lab_check.json").exists() else None
    # Отслеживание типов (MONIC) и доля маркетплейсов по типам по годам (scripts/space_dynamics.py)
    mc = sd["dynamics"].get("monic")
    monic = None
    if mc:
        yrs = sorted(mc["profile_by_year"])
        mpc = cfg["retail"]["marketplace_category"]
        monic = {"match": mc["thresholds"]["match"], "n_steps": mc["n_steps"], "overlap_min": mc["overlap_min"],
                 "other": sum(v for k, v in mc["events"].items() if k != "survive"), "years": [yrs[0], yrs[-1]],
                 "mp": {c: [mc["profile_by_year"][yrs[0]][c][mpc], mc["profile_by_year"][yrs[-1]][c][mpc]]
                        for c in mc["profile_by_year"][yrs[0]]},
                 # изменение доли каждой категории за два года: [наименьшее, наибольшее] по типам — чем выделяются маркетплейсы
                 "mp_cat": mpc,
                 "cat": {k: [round(min(v), 4), round(max(v), 4)] for k, v in (
                     (k, [mc["profile_by_year"][yrs[-1]][c][k] - mc["profile_by_year"][yrs[0]][c][k] for c in mc["profile_by_year"][yrs[0]]])
                     for k in cfg["panel"]["categories"] + [cfg["panel"]["other_category"]])}}
    wr = ag["workplace_vs_residence"]
    ka = j(o2 / "k_at_alpha.json") if (o2 / "k_at_alpha.json").exists() else None
    D = {
        "meta": {"author": cfg["report"]["author"], "K": int(len(types)), "alpha": sweep["chosen_alpha"], "seed": cfg["seed"],
                 "beta": dyn["chosen_beta"], "months": months, "alphas": [float(c[1:]) for c in a_cols],
                 "alluvial": cfg["report"]["alluvial_months"],
                 "n_mo": len(ids), "regions": regions, "profile_year": year, "cats": features.share_cols(),
                 "twins_year": cfg["twins"]["profile_year"],
                 # средние по всем МО — база отклонений в паспорте, как у типов (src/mo/interpret.py: mirkin)
                 "avg_spend": round(float(spend.mean()), 1), "hl": cfg["landing"]["mirkin_highlight"],
                 "tau": j(o2 / "networks.json")["physical_info"]["tau_km"], "profile_year_prev": str(int(year) - 1),
                 "regions_total": int(lat.loc[lat["year_to"] == 9999, "region_name"].nunique()),
                 "regions_panel": int(mo.loc[ids, "region"].nunique()), "n_gap": inv["consumption"]["territories_any_gap"],
                 "event_min": cfg["dynamics"]["event_min_months"],
                 "cite": load("data_sources")["citation"]},
        "types": types, "mo": mo_rec, "geo": {"w": geo["w"], "h": geo["h"], "views": geo["views"]},
        "agg": {"closer_ring": fc["extended_check"]["combined"]["share_closer"],
                "closer_ring_ci": fc["extended_check"]["combined"]["share_closer_ci95"],
                "closer_ctrl": fc["extended_check"]["combined"]["control_share_closer"],
                "n_pairs": fc["extended_check"]["combined"]["n_profile"],
                "n_pairs_panel": ag["workplace_vs_residence"]["ring"]["n"],
                "n_ctrl": ag["workplace_vs_residence"]["control"]["n"],
                "n_ctrl_prof": ag["profile_closeness"]["n_control"],
                "corr_ring": ag["comovement"]["mean_corr_ring"], "corr_ring_ci": ag["comovement"]["ring_ci95_mean"],
                "corr_ctrl": ag["comovement"]["mean_corr_control"], "corr_ctrl_ci": ag["comovement"]["control_ci95_mean"],
                "corr_rand": ag["comovement"]["mean_corr_random_other_region"],
                "wage_ring": ag["workplace_vs_residence"]["ring"]["corr_spend_ratio_vs_wage_ratio"],
                "wage_ring_ci": ag["workplace_vs_residence"]["ring"]["ci95"],
                "wage_ctrl": ag["workplace_vs_residence"]["control"]["corr_spend_ratio_vs_wage_ratio"],
                "wage_ctrl_ci": ag["workplace_vs_residence"]["control"]["ci95"],
                "ring_pop": ag["scale"]["ring_population_mln"], "ring_rural": 1 - ag["scale"]["ring_urban_share_weighted"],
                "ring_total": ag["scale"]["ring_pairs_total"],
                # «бублики» с данными о тратах: полная панель + расширенная проверка (как в scripts/functional_cities.py)
                "ring_n_data": ag["scale"]["ring_pairs_in_panel"] + fc["extended_check"]["n_pairs_ext"],
                # уровни: медианная зарплата и траты района в % к своему городу, доля пар с зарплатой ниже 80% городской,
                # доли пар, где траты (в % к городу) выше зарплаты и выше на 20+ п. п. (как у Иркутского района)
                "lvl": {g: [wr[g]["median_wage_pct"], wr[g]["median_spend_pct"], wr[g]["share_wage_below_80pct"],
                            wr[g]["share_spend_above_wage"], wr[g]["share_spend_above_wage_20pp"]]
                        for g in ("ring", "control")} if "median_wage_pct" in wr["ring"] else None,
                # контроль из всех соседних городов, включая другие регионы (чувствительность)
                "closer_ctrl_any": ag["profile_closeness"]["control_any_region"]["share_closer_control"]},
        "fc": {"hidden_n": fc["hidden_satellites"]["n"], "hidden_pop": fc["hidden_satellites"]["population_mln"],
               "official": fc["functional_cities"]["official_city_pop_mln"],
               "ring_pop": fc["functional_cities"]["ring_pop_mln"],
               "functional": fc["functional_cities"]["functional_pop_mln"],
               "n_cities": fc["functional_cities"]["n_cities"], "auc": fc["auc_ring_vs_control"]["index"]["auc"],
               "auc_thr": cfg["functional"]["auc_candidates_only_below"],
               "unc": fc.get("uncertainty"),
               "top": [{"n": short(x["name"]), "c": short(x["city"]), "km": x["road_km"]} for x in fc["hidden_satellites"]["top"][:12]],
               "valid": {k: [v["hidden_median"], v["other_control_median"]] for k, v in fc["hidden_satellites"]["validation_not_in_index"].items()}},
        "sweep": [{"a": r["alpha"], "SW": r["SW"], "AVI": r["AVI"], "boot": r["bootstrap_ari"], "jac": r.get("jaccard_min"),
                   "ok": r.get("stable", True), "reg": r["ari_regions"], "a0": r.get("ari_vs_alpha0")}
                  for r in sweep["rows"]],
        "thr": {"sw": (1 - cfg["sweep"]["sw_loss_max"]) * sweep["rows"][0]["SW"], "boot": cfg["methods"]["stability_ari_min"],
                "jac": cfg["methods"]["stability_jaccard_min"], "n_sub": cfg["methods"]["bootstrap"]},
        "methods": meth,
        "seeds": None if not ms else {"n": len(ms["seeds"]), "chi2": ms["friedman"]["chi2"], "p": ms["friedman"]["p_value"],
                                      "cd": ms["nemenyi_cd"], "ta": [nice[m] for m in ms["threshold_aggregation_order"]],
                                      "ref": nice[ms["reference"]], "ref_rank": ms["mean_rank"][ms["reference"]],
                                      "second": nice[second], "second_rank": ms["mean_rank"][second],
                                      "second_p": ms["wilcoxon_vs_reference_holm"][second],
                                      "second_d": ms["cohen_d_rank_diff_vs_reference"][second]},
        "panel": None if not pan else {"z": pan["methods"]["kefrin_phys"]["z"], "avi_k": pan["methods"]["kefrin_phys"]["avi_over_inv_K"]},
        "stay": sd["dynamics"]["stay_probability"], "boundary": sd["dynamics"]["boundary_mo_share"],
        "changed": dyn["final"]["changed_share_mean"], "changed_ind": dyn["independent_monthly"]["changed_share_mean"],
        "cash": {"area": {t: v["area_per_1000"] for t, v in cc["type_medians"].items()},
                 "beta": cc["regression_mp_share"]["rural"]["beta_std"]["log_area_per_1000"]["b"]},
        "mn": {"ring": mn["ring_vs_control_by_month"]["sim_ring_mean"], "ctrl": mn["ring_vs_control_by_month"]["sim_control_mean"],
               "months": mn["ring_vs_control_by_month"]["months_ring_gt_control"]},
        "twins": {"same": twj["check"]["same_type_twins"], "rand": twj["check"]["same_type_random_other_region"],
                  "wage_t": twj["check"]["median_abs_log_wage_diff_twins"], "wage_r": twj["check"]["median_abs_log_wage_diff_random"],
                  "r2": twj["region_share_of_raw_gap_variance"], "ex": twj["example_region"]},
        "rob": {"seed_min": min(rb["variants"]["seeds"]), "seed_max": max(rb["variants"]["seeds"]), "n_seeds": len(rb["variants"]["seeds"]),
                "road": rb["variants"]["road_knn_network"], "y2023": rb["variants"]["profile_2023"],
                "y2023_same": rb["variants"]["profile_2023_same_type_share"], "y2023_macro": rb["variants"]["profile_2023_macro_rural_merged"], "rural": rb["rural_types"],
                "struct": rb["variants"]["structure_only"], "no_msk": rb["variants"]["without_moscow_spb_districts"],
                "k_alt": rb["variants"]["k_alt_ari"], "alt_pair": rb.get("alt_pair")},
        "eta_wage": tj["external_validation"]["log_wage"]["eta2"],
        "pat": None if not prj else {"t1t3": prj["ordinal_patterns"]["type_pattern_hamming"]["T1-T3"],
                                     "n": prj["ordinal_patterns"]["n_pairs"]},
        # η² типа, η² базы и 95%-интервал их разницы (бутстрап МО)
        "ext": {k: [v["eta2"], v["eta2_level_groups"], v["diff_ci95"]] for k, v in tj["external_validation"].items()},
        "t1": {"n": tj["types"]["T1"]["n_mo"], "fed": tj["types"]["T1"]["n_fed_city_districts"],
               "stay_fed": sd["dynamics"]["stay_probability_split"]["T1"]["fed_cities"],
               "stay_rest": sd["dynamics"]["stay_probability_split"]["T1"]["no_fed_cities"]},
        "season": {k: sd["dynamics"]["seasonality"][k] for k in ("same_month_both_years", "expected_if_random", "agreement_by_lag")},
        "fm": None if not (ms and "friedman_seed_means" in ms) else {
            "n": ms["friedman_seed_means"]["n_blocks"], "chi2": ms["friedman_seed_means"]["chi2"],
            "p": ms["friedman_seed_means"]["p_value"], "cd": ms["friedman_seed_means"]["nemenyi_cd"],
            "rank": ms["friedman_seed_means"]["mean_rank"], "alpha": ms["cd_alpha"]},
        "leiden": {"n": cmp_["methods"]["leiden_phys"]["n_clusters"], "big": cmp_["methods"]["leiden_phys"]["sizes"][0],
                   "n_cc": n_cc},
        "netmove": netmove,
        "kalpha": None if not ka else {"K": ka["chosen_K"], "stable": ka["stable_K"], "alpha": ka["alpha"]},
        "edge": None if not er else {"nets": list(er["networks"]), "rule": {k: v["rule"] for k, v in er["networks"].items()},
                                     "alpha": {k: v["alpha"] for k, v in er["networks"].items()},
                                     "ari": {k: v["ari_final"] for k, v in er["networks"].items()},
                                     "ring": {k: v["ring_same_type"] for k, v in er["networks"].items()},
                                     "ctrl": {k: v["control_same_type"] for k, v in er["networks"].items()},
                                     "avi": {k: v["AVI_phys"] for k, v in er["networks"].items()}},
        "census": None if not cm else {"n_reg": len(cm["regions"]), "n": cm["n_matched"], "groups": cm["groups"],
                                       "tests": cm["tests"], "rho": cm["spearman"]["index_ordinary_neighbours"],
                                       "rho_all": cm["spearman"]["index_all_neighbours"]},
        "show": show,
        "monic": monic,
        "partial": None if not ptj else {
            "min": ptj["rule"]["min_months"], "year": ptj["rule"]["year"], "typed": ptj["result"]["typed"],
            "atypical": ptj["result"]["atypical"], "few": ptj["result"]["few_months"],
            "absent": ptj["coverage"]["absent_regions"], "no_data": ptj["coverage"]["mo_no_data"],
            "acc6": ptj["validation"]["random_months"][str(ptj["rule"]["min_months"])] if str(ptj["rule"]["min_months"]) in ptj["validation"]["random_months"] else None,
            "neigh": ptj["not_filled"]["neighbours_majority"], "rosstat": ptj["not_filled"]["rosstat_forest"]},
        "lab": None if not lab else {"city_types": cfg["lab"]["city_types"], "n_named": lab["named_suburbs"]["n_named"], "n_in": lab["named_suburbs"]["n_in_data"],
                                     "n_city": lab["named_suburbs"]["n_in_city_types"],
                                     "base": lab["named_suburbs"]["base"]["districts_in_city_types"], "p": lab["named_suburbs"]["p_binomial"],
                                     "rows": [{"n": short(r["name"]), "t": r["type"]} for r in lab["named_suburbs"]["rows"] if r["in_data"]],
                                     "n_nocity": sum(1 for r in lab["named_suburbs"]["rows"] if r["in_data"] and not r["city_in_data"]),
                                     "g_ring": lab["growth"]["ring"]["median_pct"], "g_city": lab["growth"]["ring_city"]["median_pct"],
                                     "g_ctrl": lab["growth"]["control"]["median_pct"]},
        "sw_loss": cfg["sweep"]["sw_loss_max"],
        # «Бублики» столиц регионов и остальных городов (проба P1) и перепись по тем же группам (разведочный разрез)
        "cap": {g: {"n": p1[k]["n_ring"], "ring": p1[k]["share_closer_ring"], "ctrl": p1[k]["share_closer_control"],
                    "p": p1[k]["perm_share_diff"]["p_value"],
                    "census": (cm["exploratory_by_partner_capital"][c]["ring_vs_control"]
                               if cm and "exploratory_by_partner_capital" in cm else None)}
                for g, k, c in (("capital", "partner_is_region_capital", "capital"),
                                ("other", "partner_not_region_capital", "not_capital"))},
        # Тип против базы «группы по уровню трат»: сколько показателей Росстата тип объясняет лучше
        "extc": {"rosstat_win": sum(1 for k, v in tj["external_validation"].items()
                                    if k not in SBER_EXT and v["diff_ci95"][0] > 0),
                 "rosstat_n": sum(1 for k in tj["external_validation"] if k not in SBER_EXT),
                 "win": sum(1 for v in tj["external_validation"].values() if v["diff_ci95"][0] > 0),
                 "n": len(tj["external_validation"])},
        "shock": shock_notes,
        # физическая сеть для слоя ребер в атласе: концы ребер, вес и координаты узлов (пиксели карты)
        "net": {"a": [int(x) for x in net["a"]], "b": [int(x) for x in net["b"]], "w": [round(float(x), 2) for x in net["w"]],
                "xy": {str(k): geo["nodes"][k] for k in sorted(set(net["a"]) | set(net["b"])) if k in geo["nodes"]}},
    }
    paths_js = json.dumps({str(k): v for k, v in geo["paths"].items()}, separators=(",", ":"))
    data_js = json.dumps(D, ensure_ascii=False, separators=(",", ":"))
    fonts = ""
    for fam, fn, rng in (("Golos Text", "golos-cyrillic", "U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116"),
                         ("Golos Text", "golos-latin", "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD"),
                         ("Onest", "onest-cyrillic", "U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116"),
                         ("Onest", "onest-latin", "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD")):
        b64 = base64.b64encode((site / "fonts" / f"{fn}.woff2").read_bytes()).decode()
        fonts += (f"@font-face{{font-family:'{fam}';font-style:normal;font-weight:400 700;font-display:swap;"
                  f"src:url(data:font/woff2;base64,{b64}) format('woff2');unicode-range:{rng}}}\n")
    html = (site / "template.html").read_text()
    n_months = len(months)
    m_word = "месяц" if n_months % 10 == 1 and n_months % 100 != 11 else "месяца" if 2 <= n_months % 10 <= 4 and not 12 <= n_months % 100 <= 14 else "месяцев"
    desc = (f"Типы локальных экономик России по тратам жителей: {len(ids):,} муниципалитетов, {n_months} {m_word} данных "
            f"СберИндекса. Пригородные районы живут как свой город, хотя статистика считает их сельскими.").replace(",", "\u00a0")
    html = html.replace("__DESC__", desc)
    sub = cfg["submission"]
    html = html.replace("__REPO__", f"https://github.com/{sub['owner']}/{sub['repo']}")
    html = html.replace("/*__FONTS__*/", fonts).replace("/*__DATA__*/null", data_js).replace("/*__PATHS__*/null", paths_js)
    (site / "index.html").write_text(html)
    for name in ("report.pdf", "report_short.pdf"):      # полный отчет и краткая версия — рядом с лендингом
        rep = ROOT / "report" / name
        if rep.exists():
            shutil.copy(rep, site / name)
    print(f"site/index.html: {len(html) / 1e6:.2f} МБ, МО на карте: {len(geo['paths'])}, в панели: {len(ids)}")


if __name__ == "__main__":
    main()
