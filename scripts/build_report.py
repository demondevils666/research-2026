"""Методологический отчет: шаблон report/report.md.j2 + числа из outputs/*.json -> report/report.md, .html, .pdf.
Краткая версия (6–8 страниц): report/report_short.md.j2 -> report/report_short.md, .html, .pdf. Общее резюме
(«Коротко», метод в шести шагах, три вопроса) — report/_summary.md.j2, подключается в оба отчета.

Ни одного числа руками: все значения подставляются из json-выходов пайплайна (фильтры n, pct, pv, ci).
Проверка «чисел без источника»: в тексте шаблона вне выражений {{ … }} ищутся числа с дробной частью
и многозначные числа, кроме годов и номеров разделов; найденное печатается для ручного просмотра.
PDF печатается Chromium (Playwright), если он установлен; иначе остаются .md и .html.
Запуск: python scripts/build_report.py [--no-pdf]
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import jinja2
import pandas as pd
import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo import numfmt  # noqa: E402
from mo.names import short  # noqa: E402
from mo.config import load, path  # noqa: E402

NBSP = " "


def n(x, d: int = 1) -> str:
    """Число по-русски: десятичная запятая, неразрывный пробел между тысячами, минус «−»."""
    if x is None:
        return "—"
    s = numfmt.fixed(x, d)
    sign, s = ("−", s[1:]) if s.startswith("-") else ("", s)
    whole, _, frac = s.partition(".")
    whole = f"{int(whole):,}".replace(",", NBSP)
    return sign + whole + ("," + frac if frac else "")


def pct(x, d: int = 0) -> str:
    return n(float(x) * 100, d) + "%"


def dpct(x, d: int = 1) -> str:
    """Разность логарифмов -> относительная разница в процентах со знаком: 0,08 -> «+8,3%»."""
    v = math.expm1(float(x)) * 100
    return ("+" if v > 0 else "") + n(v, d) + "%"


def pv(p) -> str:
    p = float(p)
    return "p < 0,001" if p < 0.001 else f"p = {n(p, 3)}"


def ci(v, d: int = 2, as_pct: bool = False) -> str:
    """Интервал: «0,67–0,84»; с отрицательной границей — «от −0,35 до −0,10», чтобы тире не путалось с минусом."""
    f = (lambda x: n(float(x) * 100, 0)) if as_pct else (lambda x: n(x, d))
    unit = "%" if as_pct else ""
    if float(v[0]) < 0:
        return f"от {f(v[0])}{unit} до {f(v[1])}{unit}"
    return f"{f(v[0])}–{f(v[1])}{unit}"


def pl(x, one: str, few: str, many: str) -> str:
    """Число с существительным в нужной форме: 1 месяц, 2 месяца, 5 месяцев."""
    k = abs(int(round(float(x))))
    form = one if k % 10 == 1 and k % 100 != 11 else few if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14 else many
    return f"{n(x, 0)} {form}"


def sci(x) -> str:
    """5.42e-14 -> 5,4·10⁻¹⁴."""
    if float(x) == 0:
        return "0"
    m, e = f"{float(x):.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m.replace('.', ',')}·10{str(int(e)).translate(sup)}"


def j(p: Path):
    return json.loads(p.read_text())


def rule_text(rule: list[dict], label: dict) -> str:
    """Правило типа словами; два условия по одному признаку (≥ a и ≤ b) склеиваются в интервал «a–b»."""
    val = lambda f, x: f"{n(x, 0)} ₽" if f == "Все категории" else f"{n(x, 1)}%"
    by = {}
    for r in rule:
        by.setdefault(r["feature"], {})[r["op"]] = r["threshold"]
    parts = []
    for f, ops in by.items():
        name = label.get(f, f)
        if ">=" in ops and "<=" in ops:
            lo, hi = val(f, ops[">="]), val(f, ops["<="])
            unit = " ₽" if lo.endswith("₽") else "%"
            parts.append(f"{name} {lo.removesuffix(unit).strip()}–{hi}")
        else:
            op = ">=" if ">=" in ops else "<="
            parts.append(f"{name} {'≥' if op == '>=' else '≤'} {val(f, ops[op])}")
    return " и ".join(parts)


def context() -> dict:
    cfg = load()
    o2, o3 = path("outputs") / "stage2", path("outputs") / "stage3"
    ctx = {"cfg": cfg, "pn": j(path("outputs") / "panel" / "summary.json"), "nw": j(o2 / "networks.json"),
           "sk": j(o2 / "select_k.json"), "sw": j(o2 / "kefrin_sweep.json"), "cmp": j(o2 / "compare.json"),
           "dyn": j(o2 / "dynamics.json"), "ty": j(o3 / "types.json"), "ag": j(o3 / "agglomerations.json"),
           "sd": j(o3 / "space_dynamics.json"), "rb": j(o3 / "robustness.json"), "fc": j(o3 / "functional_cities.json"),
           "cc": j(o3 / "cash_check.json"), "mn": j(o3 / "monthly_network.json"), "p1": j(path("outputs") / "probes" / "p1_ring_pairs.json"),
           "inv": j(path("outputs") / "inventory" / "inventory.json")}
    opt = {"panel": o2 / "icvi_panel.json", "seeds": o2 / "methods_seeds.json", "pr": o3 / "patterns_rules.json",
           "fis": o3 / "fiscal_check.json", "tw": o3 / "twins.json", "er": o2 / "edge_rules.json", "ka": o2 / "k_at_alpha.json",
           "cm": o3 / "commute_check.json", "lab": o3 / "lab_check.json", "pt": o3 / "partial_types.json"}
    for k, f in opt.items():
        ctx[k] = j(f) if f.exists() else None
    ctx["panel_perm"] = cfg["icvi_panel"]["permutations"]
    # Компоненты связности физической сети: размер и главные регионы (для описания сети и вырождения Leiden)
    import networkx as nx
    mo = pd.read_parquet(path("processed") / "mo.parquet")
    e = pd.read_parquet(path("processed") / "net_physical.parquet")
    g = nx.Graph()
    g.add_nodes_from(mo.index)
    g.add_edges_from(zip(e["a"], e["b"]))
    ctx["phys_cc"] = [{"n": len(c), "regions": mo.loc[sorted(c), "region"].value_counts().index.tolist()}
                      for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    # Что сделала сеть: МО, сменившие тип между α = 0 и выбранным α, и их связи внутри своего типа до и после
    from mo import methods as mth
    lsw = pd.read_parquet(path("processed") / "labels_sweep.parquet")
    col = {round(float(c[1:]), 4): c for c in lsw.columns}
    ref = pd.read_csv(o2 / "labels_static.csv", index_col="territory_id")["type"].loc[lsw.index].to_numpy()
    ctx["netmove"] = mth.moved_by_network(mth.align_labels(ref, lsw[col[0.0]].to_numpy()),
                                          mth.align_labels(ref, lsw[col[round(ctx["sw"]["chosen_alpha"], 4)]].to_numpy()),
                                          list(lsw.index), e)
    # Сколько разных городов у контрольных пар (пары с общим городом не независимы)
    from mo import agglo
    u = agglo.units(mo, pd.read_parquet(path("processed") / "ring_pairs.parquet"),
                    pd.read_parquet(path("processed") / "edges_contiguity.parquet"))
    uc = u[u["group"] == "control"]
    ctx["ctrl_cities"] = {"n_pairs": len(uc), "n_cities": int(uc["partner"].nunique()),
                          "n_regions": int(mo.loc[uc["unit"], "region_code"].nunique())}
    ctx["src"] = load("data_sources")
    f = ctx["fis"]
    ctx["fis_ok"] = bool(f and f["pooled"]["perm_gap_ring_minus_control"]["p_value"] < 0.05
                         and f["pooled"]["gap_mean_ring"] > f["pooled"]["gap_mean_control"])
    for name in ("verification", "canus_tuning"):
        f = o2 / f"{name}.json"
        ctx[name[:3] if name == "verification" else "ct"] = j(f) if f.exists() else None
    ctx["types_cfg"] = yaml.safe_load((ROOT / "configs" / "types.yaml").read_text())["types"]
    # «Четвертая Россия» Зубаревич: какие из слаборазвитых республик есть в полной панели и в каких типах их МО
    mt = pd.read_csv(o3 / "mo_types.csv")
    fourth = {}
    for reg in cfg["report"]["fourth_russia"]:
        x = mt[mt["region"] == reg]
        fourth[reg] = {"n": len(x), "types": ", ".join(f"{k} — {v}" for k, v in x["type"].value_counts().sort_index().items())}
    ctx["fourth"] = fourth
    # Таблица методов по среднему рангу шести ICVI конкурса
    nice = {"kefrin_phys": "KEFRiN, физическая сеть", "kefrin_sim": "KEFRiN, сеть сходства", "kmeans": "k-means",
            "gmm": "GMM", "spectral_sim": "спектральная, граф сходства", "leiden_phys": "Leiden, физическая сеть",
            "leiden_sim": "Leiden, сеть сходства", "canus_phys": "CANUS, физическая сеть"}
    rows = []
    for m, r in ctx["cmp"]["methods"].items():
        if "indices" not in r:
            continue
        ind, z = r["indices"], r["z_vs_random"]
        rows.append({"key": m, "name": nice.get(m, m), "uses": re.sub(r"(\d)\.(\d)", r"\1,\2", r["uses"]),
                     "rank": r["mean_rank_contest_icvi"],
                     "SW": ind["SW"], "CH": ind["CH"], "S_Dbw": ind["S_Dbw"], "AVI": ind["AVI@phys"], "AVU": ind["AVU@phys"],
                     "Q": ind["Q@phys"], "zSW": z["SW"], "zAVI": z["AVI@phys"], "zQ": z["Q@phys"],
                     "boot": r.get("bootstrap_ari"), "ring": r["ring_same_type"], "ctrl": r["control_same_type"],
                     "ari_reg": r["ari_regions"]})
    # Основной ранг — среднее по seed (как тест Фридмана, рис. 8 и лендинг); ранг итогового прогона — справочно
    seed_rank = (ctx.get("seeds") or {}).get("friedman_seed_means", {}).get("mean_rank", {})
    for r in rows:
        r["rank_seed"] = seed_rank.get(r["key"], r["rank"])
    ctx["methods"] = sorted(rows, key=lambda x: x["rank_seed"])
    ctx["nice_names"] = nice
    # Где метод хуже случайного разбиения: z против перестановок с «неправильным» знаком (S_Dbw и AVU — чем меньше, тем лучше)
    worse = []
    if ctx.get("panel"):
        for m, r in ctx["panel"]["methods"].items():
            if m == "leiden_phys":
                continue
            for ix, low_better in (("SW", False), ("CH_N", False), ("S_Dbw", True), ("AVI", False), ("AVU", True), ("Q", False)):
                z = r["z"][ix]
                if (z > 0) if low_better else (z < 0):
                    worse.append({"name": nice.get(m, m), "ix": {"CH_N": "CH/N", "Q": "MQ"}.get(ix, ix), "z": z})
    ctx["worse_than_random"] = worse
    ctx["best_method"] = ctx["methods"][0]["key"]
    # Развертка α: строки для таблицы (выбранный α и крайнее значение — всегда)
    keep = {0.0, 0.3, 0.5, 0.7, 0.75, 0.8, 0.9, ctx["sw"]["chosen_alpha"], ctx["sw"]["rows"][-1]["alpha"]}
    ctx["sweep_rows"] = [r for r in ctx["sw"]["rows"] if r["alpha"] in keep]
    # Отличительные черты типов: три самых больших отклонения по правилу Миркина
    label = {"Продовольствие": "продукты", "Здоровье": "здоровье", "Маркетплейсы": "маркетплейсы",
             "Общественное питание": "общепит", "Транспорт": "транспорт", "Прочее": "прочее", "Все категории": "траты"}
    ctx_lab = {"population": "население", "urban_share": "доля горожан", "wage": "зарплата", "emp_agri": "занятость в с/х",
               "emp_mining": "добыча", "emp_industry": "промышленность", "emp_market_services": "рыночные услуги",
               "emp_public": "бюджетный сектор", "market_access": "доступность рынков"}
    for code, t in ctx["ty"]["types"].items():
        dev = {**{label[k]: v for k, v in t["mirkin_shares_pct"].items()},
               "траты": t["mirkin_spend_per_capita_pct"]["Все категории"]}
        top = sorted(dev.items(), key=lambda kv: -abs(kv[1]))[:3]
        t["top_spend"] = ", ".join(f"{k} {'+' if v > 0 else '−'}{n(abs(v), 0)}%" for k, v in top)
        cdev = {ctx_lab[k]: v for k, v in t["mirkin_context_pct"].items() if k in ctx_lab}
        topc = sorted(cdev.items(), key=lambda kv: -abs(kv[1]))[:2]
        t["top_ctx"] = ", ".join(f"{k} {'+' if v > 0 else '−'}{n(abs(v), 0)}%" for k, v in topc)
        t["rule_text"] = rule_text(t["rule"]["rule"], label)
    ev = ctx["ty"]["external_validation"]
    ext_lab = {"log_wage": "зарплата (Росстат, лог)", "mobility_km": "индекс мобильности (только Северо-Запад)",
               "log_population": "население (лог)", "market_access": "доступность рынков (СберИндекс)",
               "urban_share": "доля горожан", "emp_market_services": "занятость в рыночных услугах",
               "emp_agri": "занятость в сельском хозяйстве", "emp_industry": "занятость в промышленности",
               "emp_mining": "занятость в добыче", "emp_public": "занятость в бюджетном секторе"}
    ctx["ext_rows"] = sorted(({"name": ext_lab[k], **v} for k, v in ev.items()), key=lambda x: -x["eta2"])
    # Тип против базы по всем показателям (в винительном падеже для фразы «… они объясняют лучше / не лучше»)
    acc = {"log_population": "население", "market_access": "доступность рынков", "mobility_km": "мобильность жителей",
           "emp_mining": "занятость в добыче", "emp_industry": "занятость в промышленности",
           "emp_market_services": "занятость в рыночных услугах", "log_wage": "зарплату", "urban_share": "долю горожан",
           "emp_agri": "занятость в сельском хозяйстве", "emp_public": "занятость в бюджетном секторе"}
    sber = {"market_access", "mobility_km"}                       # индексы СберИндекса, остальное — Росстат
    # сильнее / на уровне / слабее базы — по 95%-интервалу разницы η² (бутстрап МО, scripts/interpret_types.py)
    win = [k for k, v in ev.items() if v["diff_ci95"][0] > 0]
    lose = [k for k, v in ev.items() if v["diff_ci95"][1] < 0]
    tie = [k for k in ev if k not in win and k not in lose]
    nom = {**acc, "log_wage": "зарплата", "urban_share": "доля горожан"}  # именительный падеж
    row = lambda k: {"name": acc[k], "nom": nom[k], "eta2": ev[k]["eta2"], "base": ev[k]["eta2_level_groups"], "ci": ev[k]["diff_ci95"]}  # noqa: E731
    ctx["ext_sum"] = {"win": [nom[k] for k in win], "tie": [nom[k] for k in tie], "lose": [nom[k] for k in lose],
                      "win_rows": [row(k) for k in win], "tie_rows": [row(k) for k in tie], "lose_rows": [row(k) for k in lose],
                      "rosstat_win": sum(k not in sber for k in win), "rosstat_lose": sum(k not in sber for k in lose),
                      "rosstat_n": sum(k not in sber for k in ev)}
    # Ссылки на лендинг и репозиторий подачи
    sub = cfg["submission"]
    ctx["links"] = {"landing": f"https://{sub['owner']}.github.io/{sub['repo']}/", "repo": f"https://github.com/{sub['owner']}/{sub['repo']}"}
    # Известные шоки 2024 года среди МО: годовой тип против самого частого помесячного (labels_monthly.csv)
    shocks = cfg["report"].get("shocks_2024", {})
    lm = pd.read_csv(o2 / "labels_monthly.csv", index_col="territory_id")
    st = pd.read_csv(o2 / "labels_static.csv", index_col="territory_id")["type"]
    code_of = pd.crosstab(st, mt.set_index("territory_id")["type"].reindex(st.index)).idxmax(axis=1).to_dict()
    notes = []
    for _, r in mt[mt["name"].isin(shocks)].iterrows():
        months = lm.loc[r["territory_id"]].map(code_of)
        modal = months.value_counts()
        notes.append({"name": r["name"], "type": r["type"], "modal": modal.index[0], "modal_months": int(modal.iloc[0]),
                      "n_months": int(len(months)), "shock": shocks[r["name"]]})
    ctx["shock_notes"] = notes
    ctx["shock_names"] = shocks
    # Переходы типов для диаграммы (рис. 10): месяцы-столбцы из конфига, тот же расчет, что в make_figures.py
    lc = lm.apply(lambda c: c.map(code_of))
    am = cfg["report"]["alluvial_months"]
    rural = set(load("types")["rural"])
    a, b = lc[am[0]], lc[am[-1]]
    chg = a != b
    jan, jul = [m for m in am if m.endswith("-01")], [m for m in am if m.endswith("-07")]
    size = {m: lc[m].value_counts() for m in am}
    sc = max(sorted(set(lc[am[0]])), key=lambda k: abs(sum(size[m].get(k, 0) for m in jan) / max(len(jan), 1)
                                                       - sum(size[m].get(k, 0) for m in jul) / max(len(jul), 1)))
    ctx["alluvial"] = {"months": am, "same_share": float((~chg).mean()), "n_changed": int(chg.sum()),
                       "rural_share": float((a[chg].isin(rural) & b[chg].isin(rural)).mean()) if chg.any() else 0.0,
                       "rural": sorted(rural), "season_type": sc,
                       "season_jan": [int(size[m].get(sc, 0)) for m in jan], "season_jul": [int(size[m].get(sc, 0)) for m in jul]}
    return ctx


def lint(template: str) -> list[str]:
    """Числа в тексте шаблона вне {{ … }} и {% … %}: кандидаты на «число без источника»."""
    text = re.sub(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", " ", template, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)|\]\([^)]*\)", " ", text)
    found = []
    for m in re.finditer(r"(?<![\w.])(\d+[.,]\d+|\d{3,})(?![\w])", text):
        tok = m.group(0)
        if re.fullmatch(r"(19|20)\d\d", tok):
            continue
        found.append(tok)
    return found


def print_pdf(pairs: list[tuple[Path, Path]]) -> None:
    """HTML -> PDF через Chromium (Playwright): A4, номера страниц внизу."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:  # браузер другой версии Playwright: берем установленный Chromium напрямую
            import glob
            import os
            exe = os.environ.get("CHROMIUM_PATH") or next(iter(sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))), None)
            if not exe:
                raise
            b = p.chromium.launch(executable_path=exe)
        for html, pdf in pairs:
            pg = b.new_page()
            pg.goto(html.resolve().as_uri())
            pg.wait_for_load_state("networkidle")
            pg.pdf(path=str(pdf), format="A4", print_background=True, display_header_footer=True,
                   header_template="<span></span>",
                   footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#8a8984">'
                                   '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                   margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
            pg.close()
        b.close()


# Полный отчет и краткая версия: общий контекст и фильтры, общее резюме — report/_summary.md.j2
DOCS = [("report", "методологический отчет"), ("report_short", "краткая версия отчета")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    rep = ROOT / "report"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(rep), undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True)
    env.filters.update({"n": n, "pct": pct, "dpct": dpct, "pv": pv, "ci": ci, "sci": sci, "pl": pl, "short": short,
                        "dot": lambda s: s if s.endswith(".") else s + ".",
                        "sn": lambda x, d=1: ("+" if float(x) > 0 else "") + n(x, d),
                        "month_ru": lambda m: ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре",
                                               "октябре", "ноябре", "декабре"][int(m[5:7]) - 1] + f" {m[:4]}"})
    ctx = context()
    wrapper = (rep / "report.html.j2").read_text()
    pairs = []
    for name, kind in DOCS:
        src = (rep / f"{name}.md.j2").read_text()
        md = env.get_template(f"{name}.md.j2").render(**ctx)
        (rep / f"{name}.md").write_text(md)
        body = MarkdownIt("commonmark", {"html": True}).enable("table").render(md)
        html = wrapper.replace("методологический отчет</title>", f"{kind}</title>").replace("{{ body }}", body)
        (rep / f"{name}.html").write_text(html)
        sus = lint(src + ((rep / "_summary.md.j2").read_text() if name == "report" else ""))
        print(f"{name}.md: {len(md)} знаков; числа в тексте шаблона без источника: {sorted(set(sus)) or 'нет'}")
        pairs.append((rep / f"{name}.html", rep / f"{name}.pdf"))
    if args.no_pdf:
        return
    try:
        print_pdf(pairs)
        print("PDF готовы:", ", ".join(p.name for _, p in pairs))
    except Exception as ex:  # нет Playwright или Chromium — PDF пропускаем
        print("PDF не построен:", repr(ex)[:200])


if __name__ == "__main__":
    main()
