"""Рисунки для отчета и лендинга (PNG, светлая тема) — из готовых json/csv, без пересчета моделей.

fig1_types_map.png      — 6 мини-карт: где лежит каждый тип (пояс или архипелаг)
fig2_type_profiles.png  — профили типов по правилу Миркина (отклонение от среднего по стране, %)
fig3_alpha_sweep.png    — главный эксперимент: вес сети α в KEFRiN против связности, компактности и устойчивости
fig4_agglomerations.png — «город больше своих границ»: три независимых измерения
fig6_dynamics.png       — доля сменивших тип по месяцам и вероятность остаться в типе
fig7_functional_cities.png — функциональные города: город, его «бублик» и скрытые спутники (6 городов от 250 тыс.
                          с наибольшим приростом населения по тратам; выбор по данным, а не вручную)
fig8_monthly_network.png — сеть по месяцам: сходство «бублик — город» против «обычный район — город»; доля связей
                          между разными типами
fig9_methods_cd.png     — средние ранги методов по шести индексам (значения усреднены по seed) и критическая разность
                          Немени (Demšar 2006)
fig10_three_lenses.png  — витрина (commute.showcase): район вокруг города глазами Росстата, трат и переписи 2010
fig11_census_check.png  — перепись 2010: доля работающих в другом населенном пункте по группам МО
fig12_network.png       — сама физическая сеть на карте: ребра внутри типов цветом типа, между типами — серым; вся
                          страна и окрестности Иркутска (толщина ребра — вес)
fig13_transitions.png   — аллювиальная диаграмма: типы МО в выбранные месяцы (report.alluvial_months) и потоки между ними
Цвета типов — configs/types.yaml (палитра проверена валидатором); подписи типов на каждом рисунке, поэтому
идентичность не держится на одном цвете.
Размеры — под печать: ширина рисунка W = 7 дюймов (колонка PDF — 6,85 дюйма), поэтому шрифт печатается почти своим
размером (не мельче 7 pt). Длинные заголовки и сноски переносятся по ширине (title, foot), чтобы не расширять PNG.
Запуск: python scripts/make_figures.py
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import data, interpret, numfmt  # noqa: E402
from mo.config import load, path  # noqa: E402
from mo.names import short  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BASE_FILL, NODATA = "#e2e1dc", "#f3f2ef"
ACCENT, NEUTRAL = "#2a78d6", "#b9b8b2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "axes.axisbelow": True,
})
W = 7.0                  # ширина рисунка, дюймы
TITLE, FOOT = 10, 7.5    # заголовок рисунка и сноска, pt
ALBERS = "+proj=aea +lat_0=0 +lon_0=100 +lat_1=50 +lat_2=70 +ellps=WGS84 +units=m"


def ru(x: float, nd: int = 1) -> str:
    """Число с десятичной запятой (округление как в отчете и на лендинге, src/mo/numfmt.py)."""
    return numfmt.fixed(x, nd).replace(".", ",")


def plural(k: int, one: str, few: str, many: str) -> str:
    k = abs(int(k))
    form = one if k % 10 == 1 and k % 100 != 11 else few if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14 else many
    return f"{k} {form}"


def wrap(text: str, fs: float = FOOT, width_in: float = W - 0.2, em: float = 0.6) -> str:
    """Перенос по ширине рисунка: у DejaVu Sans знак в среднем ~0,6 кегля (жирный — ~0,68)."""
    n = max(20, int(width_in * 72 / (fs * em)))
    return "\n".join(textwrap.fill(p, n) for p in text.split("\n"))


def title(fig, text: str) -> None:
    """Заголовок-вывод над рисунком (выше всех осей; bbox_inches="tight" включает его в PNG)."""
    fig.text(0.01, 1.01, wrap(text, TITLE, em=0.68), ha="left", va="bottom", fontsize=TITLE, fontweight="bold", color=INK,
             linespacing=1.25)


def foot(fig, text: str, y: float = 0.0) -> None:
    """Сноска под рисунком с переносом по ширине."""
    fig.text(0.01, y, wrap(text), ha="left", va="top", fontsize=FOOT, color=MUTED, linespacing=1.35)


def save(fig, out: Path, name: str) -> None:
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    comma = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",").replace("-", "−"))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(comma)
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig_map(out, mo, code, types, inv, tj):
    g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last")
    g = g[g["year_to"] == 9999].copy()
    g["geometry"] = g.geometry.simplify(0.02)
    g = g.to_crs(ALBERS)
    g["type"] = g["territory_id"].map(code)
    n = len(types)
    rows = -(-n // 3)                                   # три столбца: карта не занимает целую страницу
    fig, axes = plt.subplots(rows, 3, figsize=(W, 1.75 * rows + 0.2))
    axes = axes.ravel()
    for ax, t in zip(axes, types.itertuples()):
        g[g["type"].isna()].plot(ax=ax, color=NODATA, linewidth=0)
        g[g["type"].notna() & (g["type"] != t.code)].plot(ax=ax, color=BASE_FILL, linewidth=0)
        g[g["type"] == t.code].plot(ax=ax, color=t.color, linewidth=0)
        k = int((code == t.code).sum())
        pop = mo.loc[code.index[code == t.code], "population"].sum() / 1e6
        ax.set_title(wrap(f"{t.code} · {t.name}", 8.5, W / 3 - 0.1, em=0.68), loc="left", fontsize=8.5)
        ax.text(0.0, 0.99, f"{k} МО · {ru(pop)} млн жителей", transform=ax.transAxes, color=INK2, fontsize=7.5, va="top")
        ax.set_axis_off()
    t1 = tj["types"]["T1"]
    note = (f"Цветом — МО этого типа, серым — других типов, самым светлым — нет полных данных "
            f"({len(inv['extras']['regions_absent_in_full_panel'])} регионов и {inv['consumption']['territories_any_gap']} МО "
            f"с пропусками). Районы Москвы и Петербурга на этом масштабе не видны ({t1['n_fed_city_districts']} из "
            f"{t1['n_mo']} МО типа T1).")
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    if n < len(axes):                                   # свободная ячейка — под пояснение
        axes[n].text(0.0, 0.9, wrap(note, FOOT, W / 3 - 0.1), transform=axes[n].transAxes, va="top", fontsize=FOOT,
                     color=MUTED, linespacing=1.35)
    else:
        foot(fig, note)
    title(fig, f"Где живет каждый тип локальной экономики (K = {n}, {load()['features']['profile_year']})")
    save(fig, out, "fig1_types_map.png")


def fig_profiles(out, tj, types):
    cols = [("spend", "Траты\nна жителя"), ("Продовольствие", "Продукты"), ("Общественное питание", "Общепит"),
            ("Транспорт", "Транспорт"), ("Здоровье", "Здоровье"), ("Маркетплейсы", "Маркет-\nплейсы"), ("Прочее", "Прочее")]
    M = []
    for t in types.itertuples():
        r = tj["types"][t.code]
        M.append([r["mirkin_spend_per_capita_pct"]["Все категории"]] + [r["mirkin_shares_pct"][c] for c, _ in cols[1:]])
    M = np.array(M)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.grid(False)
    im = ax.imshow(np.clip(M, -100, 100), cmap=cmap, vmin=-100, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(numfmt.fixed(M[i, j], 0))       # округление как на лендинге (половина — от нуля)
            ax.text(j, i, ("0%" if v == 0 else f"{v:+d}%".replace("-", "−")), ha="center", va="center", fontsize=8,
                    color=INK if abs(M[i, j]) < 60 else "#ffffff")
    ax.set_xticks(range(len(cols)), [c[1] for c in cols])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(types)), [f"{t.code} {t.short}" for t in types.itertuples()])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    fig.tight_layout()
    title(fig, "Чем типы отличаются от среднего МО России, %")
    foot(fig, "Правило Миркина: (среднее по типу − среднее по всем МО) / среднее по всем МО. Траты — на жителя в месяц, "
         "остальные столбцы — доли категорий в тратах. Шкала цвета обрезана на ±100%, в клетках — точные значения.")
    save(fig, out, "fig2_type_profiles.png")


def fig_sweep(out, sw):
    df = pd.DataFrame(sw["rows"])
    a0 = df.iloc[0]
    best = sw["chosen_alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), sharex=True)
    panels = [("AVI", "Связность в физической\nсети (AVI) ↑", None),
              ("SW", "Компактность по тратам\n(силуэт) ↑", (1 - load()["sweep"]["sw_loss_max"]) * a0["SW"]),
              ("jaccard_min", "Устойчивость: Жаккар\nхудшего типа ↑", load()["methods"]["stability_jaccard_min"])]
    for ax, (col, ttl, thr) in zip(axes, panels):
        ax.axvspan(best + 0.025, 1.0, color="#f0efec", zorder=0)
        ax.plot(df["alpha"], df[col], color=ACCENT, marker="o", markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
        if thr is not None:
            ax.axhline(thr, color=MUTED, linewidth=0.9)
            ax.text(0.0, thr, " порог", color=MUTED, fontsize=7.5, va="bottom")
        ax.axvline(best, color=INK2, linewidth=0.9)
        yb = df.loc[df["alpha"] == best, col].iloc[0]
        ax.annotate(f"α* = {ru(best)}: {ru(yb, 2)}", (best, yb), xytext=(-5, -12), textcoords="offset points", ha="right",
                    color=INK, fontsize=7.5, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("вес сети α")
        ax.set_xlim(-0.02, 0.97)
    cfg = load()
    fig.tight_layout(w_pad=1.0)
    title(fig, f"Сколько географии подмешивать в типологию: после α ≈ {ru(best)} типы перестают воспроизводиться")
    foot(fig, f"KEFRiN, K = {len(df['sizes'].iloc[0])}; α = 0 — только траты, α = 1 — только сеть. Серая зона — α, где "
         f"нарушено хотя бы одно условие правила выбора: на {cfg['methods']['bootstrap']} подвыборках медианный ARI ≥ "
         f"{ru(cfg['methods']['stability_ari_min'])} и Жаккар каждого типа ≥ {ru(cfg['methods']['stability_jaccard_min'], 2)} (как при "
         f"выборе K), потеря силуэта ≤ {cfg['sweep']['sw_loss_max']:.0%}, нет мелких кластеров и регионализации.")
    save(fig, out, "fig3_alpha_sweep.png")


def fig_agglo(out, ag):
    pc, cm, wr = ag["profile_closeness"], ag["comovement"], ag["workplace_vs_residence"]
    panels = [
        ("1. Профиль трат ближе к своему городу, чем к районам региона",
         [("«Бублик» — его город", pc["share_closer_ring"], pc["share_closer_ring_ci95"]),
          ("Обычный район — соседний город", pc["share_closer_control"], pc["share_closer_control_ci95"])],
         lambda v: ru(v * 100, 0) + "%"),
        ("2. Месячные изменения трат синхронны с городом (корреляция)",
         [("«Бублик» — его город", cm["mean_corr_ring"], cm["ring_ci95_mean"]),
          ("Обычный район — соседний город", cm["mean_corr_control"], cm["control_ci95_mean"]),
          ("Район — чужой город", cm["mean_corr_random_other_region"], None)], lambda v: ru(v, 2)),
        ("3. Связь трат с местной зарплатой (корреляция район/город)",
         [("«Бублик» — его город", wr["ring"]["corr_spend_ratio_vs_wage_ratio"], wr["ring"]["ci95"]),
          ("Обычный район — соседний город", wr["control"]["corr_spend_ratio_vs_wage_ratio"], wr["control"]["ci95"])],
         lambda v: ("+" if v > 0 else "") + ru(v, 2)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(W, 4.3), gridspec_kw={"height_ratios": [2, 3, 2]})
    for ax, (ttl, bars, fmt) in zip(axes, panels):
        labels = [b[0] for b in bars][::-1]
        vals = [b[1] for b in bars][::-1]
        cis = [b[2] for b in bars][::-1]
        colors = [ACCENT if "Бублик" in lab else NEUTRAL for lab in labels]
        y = np.arange(len(bars))
        ax.barh(y, vals, height=0.5, color=colors)
        for yi, v, ci in zip(y, vals, cis):
            if ci is not None:
                ax.plot(ci, [yi, yi], color=INK2, linewidth=1.1)
            txt = fmt(v).replace("-", "−")   # округление как в тексте отчета (numfmt)
            ax.text(max(v, ci[1] if ci else v, 0) + 0.02, yi, txt, va="center", color=INK, fontsize=8, fontweight="bold")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_title(ttl, loc="left")
        lo = min(0, min(v for v in vals) - 0.1, min((c[0] for c in cis if c), default=0) - 0.15)
        ax.set_xlim(lo, 1.0)
    fig.tight_layout(h_pad=0.8)
    title(fig, "Город больше своих границ: пригородный район тратит как свой город, хотя по статистике — село")
    sc = ag["scale"]
    pmax = max(pc["perm_share_diff"]["p_value"], cm["perm_ring_vs_control"]["p_value"],
               wr["perm_diff_ring_minus_control"]["p_value"])
    foot(fig, f"«Бублик» — район вокруг города с общим административным центром (в полной панели {sc['ring_pairs_in_panel']} пар: "
         f"профиль — по {pc['n_ring']}, синхронность и зарплата — по {wr['ring']['n']}; всего {sc['ring_pairs_total']}, в них живет "
         f"{ru(sc['ring_population_mln'])} млн человек, {ru((1 - sc['ring_urban_share_weighted']) * 100, 0)}% — сельское население по "
         f"Росстату). Отрезки — 95%-интервалы (бутстрап). "
         f"Различия «бублик против обычного района» — перестановочный тест, наибольшее p = {ru(pmax, 3)}.")
    save(fig, out, "fig4_agglomerations.png")


def fig_dynamics(out, dyn, sd, types):
    ev = dyn["final"]["changed_share_by_month"]
    months = list(ev)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(months))
    ind = dyn["independent_monthly"]["changed_share_mean"] * 100
    a1.plot(x, [ev[m] * 100 for m in months], color=ACCENT, label=f"эволюционный KEFRiN (β = {ru(dyn['chosen_beta'])})")
    a1.axhline(ind, color=NEUTRAL, linewidth=1.6, label="независимо каждый месяц (среднее)")
    a1.set_xticks(x[::3], [m[2:] for m in months][::3])
    a1.set_ylim(0, ind * 1.6)
    a1.set_ylabel("% МО, сменивших тип за месяц")
    a1.set_title("Сглаживание историей\nубирает дрожание типов", loc="left")
    a1.legend(frameon=False, loc="upper left", fontsize=7.5)
    stay = sd["dynamics"]["stay_probability"]
    tt = list(types.itertuples())[::-1]
    y = np.arange(len(tt))
    a2.barh(y, [stay[t.code] * 100 for t in tt], height=0.55, color=[t.color for t in tt])
    for yi, t in zip(y, tt):
        a2.text(stay[t.code] * 100 + 0.2, yi, f"{stay[t.code]:.1%}".replace(".", ","), va="center", fontsize=7.5, color=INK)
    a2.set_yticks(y, [f"{t.code} {t.short}" for t in tt])
    a2.tick_params(axis="y", length=0)
    a2.grid(axis="y", visible=False)
    a2.set_xlim(85, 102)
    a2.set_xlabel("вероятность остаться в типе за месяц, %")
    a2.set_title("Ядро стабильно,\nколеблются сельские типы", loc="left")
    fig.tight_layout(w_pad=1.2)
    title(fig, "Динамика: устойчивое ядро и подвижная граница")
    save(fig, out, "fig6_dynamics.png")


def fig_functional(out, fc):
    """6 крупнейших городов, у которых есть скрытые спутники: город, «бублики», спутники, остальные районы."""
    from mo import links
    sat = pd.read_csv(path("outputs") / "stage3" / "satellites.csv", index_col="territory_id")
    hid = sat[sat["hidden"]]
    b = data.borders()
    rp_all = links.ring_pairs(b[b["year_to"] == 9999])
    g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last")
    g = g[g["year_to"] == 9999].set_index("territory_id").to_crs(ALBERS)
    names = data.borders_latest(b)["municipal_district_name"]
    top = fc["functional_cities"]["largest_added_city_over_250k"][:6]
    cities = [c["city"] for c in top]
    name2id = {v: k for k, v in names.items()}
    COL = {"city": "#1f1f1f", "ring": ACCENT, "hidden": "#eb6834"}
    fig, axes = plt.subplots(2, 3, figsize=(W, 5.6))
    for ax, cname in zip(axes.ravel(), cities):
        cid = name2id[cname]
        ring = rp_all.loc[rp_all["city"] == cid, "ring"].tolist()
        hs = hid.index[hid["partner"] == cid].tolist()
        units = [cid] + ring + hs
        x0, y0, x1, y1 = g.loc[[u for u in units if u in g.index]].total_bounds
        # квадратное окно с полями: все панели одного вида
        cx_, cy_, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.65 + 5e3
        x0, x1, y0, y1 = cx_ - half, cx_ + half, cy_ - half, cy_ + half
        box = g.cx[x0:x1, y0:y1]
        box.plot(ax=ax, color=BASE_FILL, edgecolor=SURFACE, linewidth=0.5)
        for kind, ids_ in (("ring", ring), ("hidden", hs), ("city", [cid])):
            sub = g.loc[[u for u in ids_ if u in g.index]]
            if len(sub):
                sub.plot(ax=ax, color=COL[kind], edgecolor=SURFACE, linewidth=0.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_axis_off()
        r = next(c for c in top if c["city"] == cname)
        ax.set_title(r["center"], loc="left", fontsize=9)
        parts = [f"{ru(r['city_pop'] / 1e6, 2)} млн → {ru(r['functional_pop'] / 1e6, 2)} млн по тратам"]
        if ring:
            parts.append("«бублик»: " + ", ".join(short(names[t]) for t in ring))
        if hs:
            parts.append("кандидат в спутники: " + ", ".join(short(n) for n in sat.loc[hs, "name"]))
        ax.text(0.0, -0.02, wrap("\n".join(parts), 7, W / 3 - 0.15), transform=ax.transAxes, color=INK2, fontsize=7,
                va="top", linespacing=1.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COL["city"], COL["ring"], COL["hidden"], BASE_FILL)]
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.2)
    fig.legend(handles, ["город", "«бублик» (общий центр с городом)", "кандидат в скрытые спутники (по тратам)",
                         "остальные МО"], loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=7.5)
    f = fc["functional_cities"]
    title(fig, "Город больше своих границ: районы, которые по тратам живут как город")
    ex = fc["extended_check"]["combined"]
    foot(fig, f"«Бублики» в среднем живут как свой город ({ex['share_closer']:.0%} из {ex['n_profile']} ближе к городу, чем "
         f"к районам региона). Кандидаты в скрытые спутники — обычные районы с индексом спутника не ниже медианы «бубликов» "
         f"({fc['hidden_satellites']['n']} районов, {ru(fc['hidden_satellites']['population_mln'])} млн жителей; перепись их "
         f"не подтвердила). По {f['n_cities']} городам: официально {ru(f['official_city_pop_mln'])} млн, с «бубликами» "
         f"{ru(f['official_city_pop_mln'] + f['ring_pop_mln'])} млн, с кандидатами — до {ru(f['functional_pop_mln'])} млн. "
         f"Показаны 6 городов от 250 тыс. жителей с наибольшим приростом по тратам; под картой — численность официально "
         f"и вместе с «бубликом» и кандидатами.", y=-0.01)
    save(fig, out, "fig7_functional_cities.png")


def fig_monthly(out, mn):
    tab = pd.read_csv(path("outputs") / "stage3" / "monthly_network.csv", index_col="date")
    x = np.arange(len(tab))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.7))
    a1.plot(x, tab["sim_ring_city"], color=ACCENT, label="«бублик» — его город")
    a1.plot(x, tab["sim_control_city"], color=NEUTRAL, label="обычный район — город")
    a1.set_xticks(x[::3], [m[2:] for m in tab.index][::3])
    a1.set_ylim(0, 1)
    a1.set_ylabel("сходство профилей в месяце (0–1)")
    a1.legend(frameon=False, loc="lower left", fontsize=7.5)
    r = mn["ring_vs_control_by_month"]
    a1.set_title(f"Пригород похож на свой город\nкаждый месяц ({r['months_ring_gt_control']} из {r['months']})", loc="left")
    a2.plot(x, tab["boundary_edge_share"] * 100, color=ACCENT)
    a2.set_xticks(x[::3], [m[2:] for m in tab.index][::3])
    a2.set_ylabel("% связей между разными типами")
    a2.set_ylim(0, 60)
    ns = mn["neighbors_similarity"]
    a2.set_title("Почти половина связей соединяет\nразные типы, доля медленно растет", loc="left")
    fig.tight_layout(w_pad=1.2)
    title(fig, "Сеть по месяцам: вес связи = близость по дороге × сходство трат в этом месяце")
    foot(fig, f"{mn['n_edges']:,}".replace(",", "\u00a0") + f" связей физической сети, {plural(len(tab), 'месяц', 'месяца', 'месяцев')}. Веса соседних месяцев "
         f"согласованы (ранговая корреляция {ru(ns['w_rank_corr_consecutive_mean'], 2)}); эволюционный KEFRiN на этой сети "
         f"дает почти те же типы (ARI с основным вариантом "
         f"{ru(mn['evolutionary_kefrin_monthly_network']['ari_with_main_by_month_mean'], 2)}).")
    save(fig, out, "fig8_monthly_network.png")


def fig_cd(out, ms):
    """Диаграмма критических разностей: методы на оси средних рангов, отрезки — группы без значимых различий."""
    nice = {"kefrin_phys": "KEFRiN · физ. сеть", "kefrin_sim": "KEFRiN · сеть сходства", "kmeans": "k-means",
            "gmm": "GMM", "spectral_sim": "Спектральная", "leiden_phys": "Leiden · физ. сеть",
            "canus_phys": "CANUS · физ. сеть", "leiden_sim": "Leiden · сеть сходства"}
    fm = ms["friedman_seed_means"]          # блоки — шесть индексов (значения усреднены по seed): независимые блоки
    r = sorted(fm["mean_rank"].items(), key=lambda kv: kv[1])
    cd = fm["nemenyi_cd"]
    k = len(r)
    fig, ax = plt.subplots(figsize=(W, 2.6))
    fig.subplots_adjust(left=0.27, right=0.73, top=0.97, bottom=0.03)   # подписи методов — в полях слева и справа
    ax.grid(False)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.5 - 0.35 * k - 0.15, 1.2)
    ax.axhline(0, color=INK2, linewidth=0.9)
    for x in range(1, k + 1):
        ax.plot([x, x], [0, 0.12], color=INK2, linewidth=0.9)
        ax.text(x, 0.3, str(x), ha="center", fontsize=8, color=INK2)
    for i, (m, v) in enumerate(r):
        y = -0.5 - 0.35 * i
        col = ACCENT if m == ms["reference"] else INK2
        ax.plot([v, v], [0, y], color=col, linewidth=0.9)
        left = v <= (k + 1) / 2
        ax.plot([v, 0.5 if left else k + 0.5], [y, y], color=col, linewidth=0.9, clip_on=False)
        ax.text(0.4 if left else k + 0.6, y, f"{nice.get(m, m)} ({ru(v, 2)})", ha="right" if left else "left",
                va="center", fontsize=8, color=INK if m != ms["reference"] else ACCENT,
                fontweight="bold" if m == ms["reference"] else "normal")
    # группы без значимых различий: максимальные отрезки с размахом рангов < CD
    vals = [v for _, v in r]
    groups, last_end = [], -1
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] < cd)
        if j > i and j > last_end:
            groups.append((vals[i], vals[j]))
            last_end = j
    for g, (a, b) in enumerate(groups):
        y = -0.25 - 0.1 * g
        ax.plot([a - 0.03, b + 0.03], [y, y], color=INK, linewidth=2.5, solid_capstyle="round")
    ax.plot([1, 1 + cd], [0.85, 0.85], color=INK, linewidth=1.6)
    ax.text(1 + cd / 2, 0.95, f"критическая разность {ru(cd, 2)}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_axis_off()
    title(fig, f"Средний ранг методов по {fm['n_blocks']} индексам конкурса (значения усреднены по {len(ms['seeds'])} seed; "
          f"1 — лучший). Тест Фридмана: χ² = {ru(fm['chi2'], 1)}, "
          f"p {'< 0,001' if fm['p_value'] < 1e-3 else '= ' + ru(fm['p_value'], 3)}")
    foot(fig, f"Черные отрезки соединяют методы, ранги которых различаются меньше критической разности Немени "
         f"(α = {ru(ms['cd_alpha'], 2)}): между ними нет значимых различий. Блоки — индексы: seed одного индекса не "
         f"независимые повторы, поэтому они усреднены.")
    save(fig, out, "fig9_methods_cd.png")


def fig_lenses(out, cm, mo):
    """Витрина (commute.showcase): район вокруг города глазами Росстата, трат и переписи — карта-локатор, три полосы
    с одной цифрой каждая и объяснение механизма. Числа — из commute_check.json (showcase)."""
    sc = cm["showcase"]
    cid, rid = sc["city_id"], sc["district_id"]
    g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last")
    g = g[g["year_to"] == 9999].set_index("territory_id").to_crs(ALBERS)
    x0, y0, x1, y1 = g.loc[[cid, rid]].total_bounds
    cx_, cy_, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.7
    box = g.cx[cx_ - half:cx_ + half, cy_ - half:cy_ + half]
    district = sc["district"].replace(" муниципальный район", " район")
    city = mo.loc[cid, "name"].replace("городской округ город ", "")
    fig = plt.figure(figsize=(W, 3.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.05], wspace=0.05)
    am, ab = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # карта-локатор: только район и город, соседи — фон
    am.grid(False)
    box.plot(ax=am, color=BASE_FILL, edgecolor=SURFACE, linewidth=0.5)
    g.loc[[rid]].plot(ax=am, color=ACCENT, edgecolor=SURFACE, linewidth=0.5)
    g.loc[[cid]].plot(ax=am, color=INK, edgecolor=SURFACE, linewidth=0.5)
    pr = g.loc[rid].geometry.buffer(-8e3).representative_point()
    pc = g.loc[cid].geometry.representative_point()
    am.annotate(district, (pr.x, pr.y), ha="center", va="center", fontsize=8, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9))
    am.annotate(city, (pc.x, pc.y), xytext=(0, 20), textcoords="offset points", ha="center", fontsize=8, color=INK,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9))
    am.set_xlim(cx_ - half, cx_ + half)
    am.set_ylim(cy_ - half, cy_ + half)
    am.set_axis_off()
    am.set_title("Где это", loc="left")
    # три полосы: одна шкала 0–100%, у каждой одна цифра и одна отметка для сравнения; в подписи — кто и как считает
    rows = [
        ("Росстат: средняя зарплата тех, кто работает в районе", f"{load()['rosstat']['year']}, % от зарплаты в городе",
         sc["wage_pct_of_city"], NEUTRAL, 100, f"{city} = 100%"),
        ("СберИндекс: траты тех, кто живет в районе", f"{sc['year']}, % от трат горожан",
         sc["spend_pct_of_city"], ACCENT, 100, f"{city} = 100%"),
        ("Перепись 2010: работают в другом населенном пункте", "% занятых жителей района",
         sc["commute_pct"], ACCENT, sc["region_median_pct"], f"типичный район области — {ru(sc['region_median_pct'], 0)}%"),
    ]
    ab.grid(False)
    for s_ in ab.spines.values():
        s_.set_visible(False)
    ab.set_xlim(0, 122)
    ab.set_ylim(-0.55, len(rows) - 0.05)
    ab.set_xticks([])
    ab.set_yticks([])
    for i, (ttl, sub, v, col, ref, ref_label) in enumerate(rows):
        y = len(rows) - 1 - i
        ab.text(0, y + 0.52, ttl, fontsize=8.5, color=INK, fontweight="bold", va="bottom")
        ab.text(0, y + 0.24, sub, fontsize=7.5, color=INK2, va="bottom")
        ab.barh(y, 100, height=0.26, color=GRID)
        ab.barh(y, v, height=0.26, color=col)
        ab.text(103, y, f"{ru(v, 0)}%", va="center", fontsize=12, color=INK, fontweight="bold")
        ab.plot([ref, ref], [y - 0.2, y + 0.2], color=INK2, linewidth=1.2)
        if ref >= 90:   # отметка «город = 100%» — над концом шкалы, на строке пояснения (оно короче шкалы)
            ab.text(ref, y + 0.24, ref_label, ha="right", va="bottom", fontsize=7.5, color=INK2)
        else:
            ab.text(ref - 1, y - 0.2, ref_label, ha="left", va="top", fontsize=7.5, color=INK2)
    ab.set_ylim(-0.55, len(rows) - 0.2)
    fig.subplots_adjust(left=0.0, right=0.97, top=0.99, bottom=0.0)
    title(fig, f"{district}: по официальной статистике — бедное село, по тратам и поездкам на работу — часть города "
          f"{city}")
    fig.text(0.01, -0.03, wrap(f"Почему так. Росстат считает «среднемесячную заработную плату работников организаций» "
                               f"(без малого бизнеса) по отчетам организаций района. Житель района, который работает в городе, "
                               f"попадает в зарплату города, а его траты — в траты района; по переписи 2010 года в другом "
                               f"населенном пункте работали {ru(sc['commute_pct'], 0)}% занятых. Поэтому по зарплате район "
                               f"выглядит бедным селом ({ru(sc['district_rural_pct'], 0)}% жителей — сельские), а по тратам — "
                               f"частью города.",
                               8, W - 0.5),
             ha="left", va="top", fontsize=8, color=INK, linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.55", fc="#eaf1fa", ec="none"))
    save(fig, out, "fig10_three_lenses.png")


def fig_network(out, mo, code, types, cm, tau):
    """Физическая сеть на карте. Узел — МО (точка внутри его границы), ребро — общая граница с весом
    w = e^(−км/τ). Ребро между МО одного типа — цветом типа, между разными типами — серым. Слева вся страна, справа —
    окрестности района-витрины (commute.showcase), толщина ребра — вес. Доля веса ребер внутри типов сравнивается
    с ожидаемой при случайной раскладке типов тех же размеров."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    e = pd.read_parquet(path("processed") / "net_physical.parquet")
    g = data.polygons().sort_values("year_to").drop_duplicates("territory_id", keep="last")
    g = g[g["year_to"] == 9999].set_index("territory_id").to_crs(ALBERS)
    g["geometry"] = g.geometry.simplify(2e3)
    pt = g.geometry.representative_point()
    color = types.set_index("code")["color"].to_dict()
    e = e[e["a"].isin(code.index) & e["b"].isin(code.index) & e["a"].isin(pt.index) & e["b"].isin(pt.index)]
    same = (code.reindex(e["a"]).to_numpy() == code.reindex(e["b"]).to_numpy())
    within = float(e.loc[same, "w"].sum() / e["w"].sum())
    nk = code.value_counts().to_numpy()
    n = int(nk.sum())
    expected = float((nk * (nk - 1)).sum() / (n * (n - 1)))
    seg = np.stack([np.c_[pt.loc[e["a"]].x, pt.loc[e["a"]].y], np.c_[pt.loc[e["b"]].x, pt.loc[e["b"]].y]], axis=1)
    ecol = np.where(same, code.reindex(e["a"]).map(color).to_numpy(), "#a3a29c")

    fig = plt.figure(figsize=(W, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.04)
    ax, az = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, az):
        a.grid(False)
        a.set_axis_off()
    # вся страна: тонкие ребра, точки-узлы
    g.plot(ax=ax, color=NODATA, linewidth=0)
    g.loc[g.index.isin(code.index)].plot(ax=ax, color=BASE_FILL, linewidth=0)
    order = np.argsort(same)                       # межтиповые ребра снизу, ребра внутри типов сверху
    ax.add_collection(LineCollection(seg[order], colors=ecol[order], linewidths=0.25 + 0.55 * e["w"].to_numpy()[order]))
    nodes = pt.loc[code.index]
    ax.scatter(nodes.x, nodes.y, s=0.5, c=code.map(color).to_numpy(), linewidths=0, zorder=3)
    ax.set_xlim(*g.total_bounds[[0, 2]])
    ax.set_ylim(*g.total_bounds[[1, 3]])
    ax.set_aspect("equal")
    ax.set_title(f"а. Вся сеть: {n:,} МО, {len(e):,} ребер".replace(",", "\u00a0"), loc="left")
    # окрестности района-витрины: заливка типом, толщина ребра — вес
    sc = cm["showcase"]
    cid, rid = sc["city_id"], sc["district_id"]
    x0, y0, x1, y1 = g.loc[[cid, rid]].total_bounds
    cx_, cy_, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.95
    box = g.cx[cx_ - half:cx_ + half, cy_ - half:cy_ + half]
    box.plot(ax=az, color=[color.get(code.get(i), NODATA) if i in code.index else NODATA for i in box.index],
             alpha=0.35, edgecolor=SURFACE, linewidth=0.6)
    inb = e["a"].isin(box.index) & e["b"].isin(box.index)
    az.add_collection(LineCollection(seg[inb.to_numpy()], colors=ecol[inb.to_numpy()],
                                     linewidths=0.6 + 2.6 * e.loc[inb, "w"].to_numpy(), capstyle="round"))
    bn = [i for i in box.index if i in code.index]
    az.scatter(pt.loc[bn].x, pt.loc[bn].y, s=14, c=[color[code[i]] for i in bn], edgecolors=SURFACE, linewidths=0.8, zorder=3)
    for tid in (cid, rid):
        nm = mo.loc[tid, "name"].replace("городской округ город ", "").replace(" муниципальный район", " район")
        az.annotate(nm, (pt[tid].x, pt[tid].y), xytext=(0, 9 if tid == cid else -12), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    az.set_xlim(cx_ - half, cx_ + half)
    az.set_ylim(cy_ - half, cy_ + half)
    az.set_aspect("equal")
    az.set_title("б. Иркутск и соседи: толщина — вес ребра", loc="left")
    handles = [Line2D([], [], color=r.color, lw=2, label=f"{r.code} {r.short}") for r in types.itertuples()]
    handles.append(Line2D([], [], color="#a3a29c", lw=2, label="между типами"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06),
               handlelength=1.4, columnspacing=1.2, fontsize=7.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.12)
    title(fig, "Сама сеть: соседи по границе, вес — близость центров по дороге")
    foot(fig, f"Ребро — общая граница, вес w = e^(−км/τ), τ = {ru(tau, 1)} км. Цвет ребра — тип, если оба МО одного типа; "
              f"серое — ребро между типами. Внутри типов — {ru(within * 100, 0)}% веса ребер; при случайной раскладке "
              f"типов тех же размеров было бы {ru(expected * 100, 0)}%.", y=-0.11)
    save(fig, out, "fig12_network.png")
    return {"within_weight_share": within, "expected_random": expected}


def fig_census(out, cm):
    """Перепись 2010: насколько чаще, чем в типичном районе своего региона, жители групп МО работают в другом населенном
    пункте (среднее отклонение доли от медианы районов региона, 95%-интервал бутстрапа)."""
    G = cm["groups"]
    rows = [("ring", "«Бублики» (район вокруг города)", ACCENT),
            ("hidden", "Кандидаты в скрытые спутники", "#eb6834"),
            ("control", "Обычные соседи города", NEUTRAL), ("other", "Остальные МО", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    ax.grid(axis="y", visible=False)
    y = np.arange(len(rows))[::-1]
    for yi, (g, lab, col) in zip(y, rows):
        r = G[g]
        ax.barh(yi, r["mean_dev_pp"], height=0.55, color=col)
        lo, hi = r["mean_dev_ci95_pp"]
        ax.plot([lo, hi], [yi, yi], color=INK, linewidth=1.1)
        ax.text(max(hi, 0) + 0.7, yi + 0.08, f"{'+' if r['mean_dev_pp'] > 0 else ''}{ru(r['mean_dev_pp'], 1)} п. п.".replace("-", "−"),
                va="center", fontsize=8.5, color=INK, fontweight="bold")
        ax.text(max(hi, 0) + 0.7, yi - 0.27, f"{plural(r['n'], 'МО', 'МО', 'МО')}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.tick_params(axis="y", length=0)
    lo_all = min(min(G[g]["mean_dev_ci95_pp"][0] for g, *_ in rows), 0)
    ax.set_xlim(lo_all - 2, max(G[g]["mean_dev_ci95_pp"][1] for g, *_ in rows) + 9)
    ax.set_xlabel("на сколько п. п. доля работающих в другом населенном пункте выше,\nчем у типичного района своего региона")
    fig.tight_layout()
    tr, th = cm["tests"]["ring_vs_control"], cm["tests"]["hidden_vs_control"]
    pv = lambda p: "p < 0,001" if p < 0.001 else f"p = {ru(p, 2)}"  # noqa: E731
    who = "из «бубликов» и скрытых спутников — да" if th["p_value"] < 0.05 else "из «бубликов» — да, из кандидатов в спутники — нет"
    title(fig, f"Перепись 2010 ({plural(len(cm['regions']), 'регион', 'региона', 'регионов')}): ездят ли на работу "
          f"в другой населенный пункт? {who[0].upper() + who[1:]}")
    foot(fig, f"Отрезок — 95%-интервал (бутстрап). «Бублики» против обычных соседей: "
         f"{'+' if tr['diff_pp'] > 0 else ''}{ru(tr['diff_pp'], 1)} п. п., {pv(tr['p_value'])}; кандидаты в спутники: "
         f"{'+' if th['diff_pp'] > 0 else ''}{ru(th['diff_pp'], 1)} п. п., {pv(th['p_value'])} (перестановки меток внутри региона).")
    save(fig, out, "fig11_census_check.png")


def month_ru(m: str) -> str:
    names = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    return f"{names[int(m[5:7]) - 1]} {m[:4]}"


def transitions(lm: pd.DataFrame, months: list, rural: set) -> dict:
    """Сводка для подписи диаграммы переходов: сколько МО в том же типе в первом и последнем столбце,
    какая доля смен — между сельскими типами."""
    a, b = lm[months[0]], lm[months[-1]]
    ch = a != b
    return {"same_share": float((~ch).mean()), "n_changed": int(ch.sum()),
            "rural_share": float((a[ch].isin(rural) & b[ch].isin(rural)).mean()) if ch.any() else 0.0}


def fig_alluvial(out, lm, types, months, rural):
    """Аллювиальная диаграмма: столбцы — месяцы, блоки — типы (высота — число МО), ленты — МО, перешедшие из типа
    в тип между соседними столбцами. Цвет ленты — тип, из которого МО ушел."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath
    tt = list(types.itertuples())
    codes = [t.code for t in tt]
    col = {t.code: t.color for t in tt}
    n = len(lm)
    gap, nw = n * 0.025, 0.07                                 # зазор между блоками (в МО) и ширина блока (доля шага)
    fig, ax = plt.subplots(figsize=(W, 4.2))
    tops = {}                                                 # (столбец, тип) -> верх блока
    for i, m in enumerate(months):
        cnt = lm[m].value_counts().reindex(codes, fill_value=0)
        y = n + gap * (len(codes) - 1)
        for c in codes:
            tops[(i, c)] = y
            ax.add_patch(Rectangle((i - nw / 2, y - cnt[c]), nw, cnt[c], color=col[c], lw=0))
            if i == 0:
                ax.text(i - nw / 2 - 0.04, y - cnt[c] / 2, f"{c} {types.set_index('code').loc[c, 'short']} · {cnt[c]}",
                        ha="right", va="center", fontsize=7.5, color=INK)
            elif i == len(months) - 1:
                ax.text(i + nw / 2 + 0.04, y - cnt[c] / 2, f"{cnt[c]}", ha="left", va="center", fontsize=7.5, color=INK)
            y -= cnt[c] + gap
        ax.text(i, n + gap * (len(codes) - 1) + n * 0.03, month_ru(m), ha="center", va="bottom", fontsize=8, color=INK2)
    for i in range(len(months) - 1):
        ct = pd.crosstab(lm[months[i]], lm[months[i + 1]]).reindex(index=codes, columns=codes, fill_value=0)
        out_y = {c: tops[(i, c)] for c in codes}              # текущий верх «выхода» из блока
        in_y = {c: tops[(i + 1, c)] for c in codes}
        x0, x1 = i + nw / 2, i + 1 - nw / 2
        xm = (x0 + x1) / 2
        for a in codes:
            for b in codes:
                v = ct.loc[a, b]
                if not v:
                    continue
                ya, yb = out_y[a], in_y[b]
                verts = [(x0, ya), (xm, ya), (xm, yb), (x1, yb), (x1, yb - v), (xm, yb - v), (xm, ya - v), (x0, ya - v), (x0, ya)]
                codes_p = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
                           MPath.CURVE4, MPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MPath(verts, codes_p), facecolor=col[a], edgecolor="none",
                                       alpha=0.35 if a == b else 0.75))
                out_y[a] -= v
                in_y[b] -= v
    ax.set_xlim(-0.95, len(months) - 1 + 0.25)
    ax.set_ylim(-n * 0.01, n * 1.18)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)
    s = transitions(lm, months, rural)
    rr = sorted(rural)
    title(fig, f"Переходы между типами: {ru(s['same_share'] * 100, 0)}% МО в {month_ru(months[-1])} в том же типе, что в "
               f"{month_ru(months[0])}; {ru(s['rural_share'] * 100, 0)}% смен — между {', '.join(rr[:-1])} и {rr[-1]} (малые города и село)")
    # сезонность размера: самый подвижный тип в январе и июле (если такие месяцы есть среди столбцов)
    jan, jul = [m for m in months if m.endswith("-01")], [m for m in months if m.endswith("-07")]
    season = ""
    if jan and jul:
        size = {m: lm[m].value_counts() for m in jan + jul}
        c = max(codes, key=lambda k: abs(np.mean([size[m].get(k, 0) for m in jan]) - np.mean([size[m].get(k, 0) for m in jul])))
        season = (f" Размер типа {c} колеблется по сезону: в январе {' и '.join(str(size[m].get(c, 0)) for m in jan)} МО, "
                  f"в июле {' и '.join(str(size[m].get(c, 0)) for m in jul)}; отдельные МО при этом меняют тип не в одни и те же "
                  f"месяцы (раздел 6.2).")
    nn = f"{n:,}".replace(",", "\u00a0")
    foot(fig, f"Высота блока — число МО типа в месяце, ленты — МО, перешедшие из типа в тип между соседними столбцами "
              f"(цвет — тип, из которого ушли; бледные — оставшиеся). Эволюционный KEFRiN, {nn} МО полной панели." + season,
         y=0.02)
    save(fig, out, "fig13_transitions.png")
    return s


def main() -> None:
    proc, o2, o3 = path("processed"), path("outputs") / "stage2", path("outputs") / "stage3"
    out = path("outputs") / "figures"
    out.mkdir(parents=True, exist_ok=True)
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(o2 / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    types = interpret.type_names(lab).sort_values("code")
    code = lab.map(types["code"])
    fig_map(out, mo, code, types, json.loads((path("outputs") / "inventory" / "inventory.json").read_text()),
            json.loads((o3 / "types.json").read_text()))
    fig_profiles(out, json.loads((o3 / "types.json").read_text()), types)
    fig_sweep(out, json.loads((o2 / "kefrin_sweep.json").read_text()))
    fig_agglo(out, json.loads((o3 / "agglomerations.json").read_text()))
    fig_dynamics(out, json.loads((o2 / "dynamics.json").read_text()), json.loads((o3 / "space_dynamics.json").read_text()), types)
    fig_functional(out, json.loads((o3 / "functional_cities.json").read_text()))
    fig_monthly(out, json.loads((o3 / "monthly_network.json").read_text()))
    ms = o2 / "methods_seeds.json"
    if ms.exists():
        fig_cd(out, json.loads(ms.read_text()))
    cm = o3 / "commute_check.json"
    if cm.exists():
        fig_lenses(out, json.loads(cm.read_text()), mo)
        fig_census(out, json.loads(cm.read_text()))
        fig_network(out, mo.loc[ids], code, types, json.loads(cm.read_text()),
                    json.loads((o2 / "networks.json").read_text())["physical_info"]["tau_km"])
    cfg = load()
    lm = pd.read_csv(o2 / "labels_monthly.csv", index_col="territory_id").loc[ids].replace(types["code"].to_dict())
    fig_alluvial(out, lm, types, cfg["report"]["alluvial_months"], set(load("types")["rural"]))


if __name__ == "__main__":
    main()
