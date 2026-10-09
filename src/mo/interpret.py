"""Интерпретация типов: названия из configs/types.yaml, профили по правилу Миркина, внешняя проверка (η²)."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from .config import load


def type_names(labels: pd.Series) -> pd.DataFrame:
    """Таблица типов из configs/types.yaml с проверкой «якорей» (МО, обязанный быть в своем типе)."""
    t = pd.DataFrame(load("types")["types"])
    for r in t.itertuples():
        got = labels.get(r.anchor)
        if got != r.label:
            raise ValueError(f"{r.code}: якорь {r.anchor} в кластере {got}, а не {r.label} — номера кластеров сменились, "
                             "обновите configs/types.yaml")
    return t.set_index("label")


def rural_codes() -> list[str]:
    """Коды сельских и малогородских типов (configs/types.yaml: rural) — для проверок онлайна в аграрных районах и надежности."""
    return list(load("types")["rural"])


def nearest_type(P: pd.DataFrame, ref: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Тип вне выборки по ближайшему центру. Профили P стандартизуются средними и разбросом эталона ref (профили МО,
    по которым строилась типология); центр типа — среднее стандартизованных профилей его МО. Возвращает метку
    ближайшего центра и евклидово расстояние до него."""
    mu, sd = ref.mean(), ref.std(ddof=0)
    cent = ((ref - mu) / sd).groupby(labels.reindex(ref.index)).mean()
    Z = ((P[ref.columns] - mu) / sd).to_numpy()
    d = np.sqrt(((Z[:, None, :] - cent.to_numpy()[None, :, :]) ** 2).sum(-1))
    return pd.DataFrame({"label": cent.index[d.argmin(1)], "distance": d.min(1)}, index=P.index)


def mirkin(values: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Правило Миркина: (среднее по типу − среднее по всем) / среднее по всем, в %. Строки — типы."""
    mu = values.mean()
    return ((values.groupby(labels).mean() - mu) / mu.abs() * 100).round(1)


def eta2(x: pd.Series, labels: pd.Series) -> dict:
    """Доля дисперсии, объясняемая типом (η²), и p-значение Краскела–Уоллиса."""
    d = pd.DataFrame({"x": x, "g": labels}).dropna()
    ss_t = ((d["x"] - d["x"].mean()) ** 2).sum()
    ss_w = d.groupby("g")["x"].apply(lambda v: ((v - v.mean()) ** 2).sum()).sum()
    groups = [g["x"].to_numpy() for _, g in d.groupby("g") if len(g) > 1]
    p = kruskal(*groups).pvalue if len(groups) > 1 else np.nan
    return {"eta2": float(1 - ss_w / ss_t) if ss_t > 0 else np.nan, "kruskal_p": float(p), "n": int(len(d))}


def _eta2_codes(x: np.ndarray, g: np.ndarray, k: int) -> float:
    n = np.bincount(g, minlength=k)
    s = np.bincount(g, weights=x, minlength=k)
    ok = n > 0
    ss_b = (s[ok] ** 2 / n[ok]).sum() - x.sum() ** 2 / len(x)
    ss_t = ((x - x.mean()) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else np.nan


def eta2_diff_ci(x: pd.Series, a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 42) -> list[float]:
    """95%-интервал разницы η²(a) − η²(b): бутстрап МО при фиксированных разбиениях a и b."""
    d = pd.DataFrame({"x": x, "a": a, "b": b}).dropna()
    xv = d["x"].to_numpy(float)
    ga, gb = pd.factorize(d["a"])[0], pd.factorize(d["b"])[0]
    ka, kb = ga.max() + 1, gb.max() + 1
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(xv), len(xv))
        diffs[i] = _eta2_codes(xv[j], ga[j], ka) - _eta2_codes(xv[j], gb[j], kb)
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def typical(Y: pd.DataFrame, labels: pd.Series, n: int = 5) -> dict:
    """Самые типичные МО каждого типа — ближайшие к центру типа в пространстве признаков."""
    out = {}
    for k, idx in labels.groupby(labels).groups.items():
        c = Y.loc[idx].mean()
        d = ((Y.loc[idx] - c) ** 2).sum(axis=1)
        out[k] = d.sort_values().index[:n].tolist()
    return out


def _round_thr(v: float, step: float) -> float:
    return float(np.round(v / step) * step)


def _atoms(X: pd.DataFrame, qs: list[float], steps: dict) -> list[tuple[str, str, float]]:
    out = set()
    for f in X.columns:
        for q in qs:
            t = _round_thr(X[f].quantile(q), steps.get(f, 0.1))
            out.add((f, ">=", t))
            out.add((f, "<=", t))
    return sorted(out)


def _mask(X: pd.DataFrame, rule: tuple) -> np.ndarray:
    m = np.ones(len(X), dtype=bool)
    for f, op, t in rule:
        m &= (X[f].to_numpy() >= t) if op == ">=" else (X[f].to_numpy() <= t)
    return m


def _f1(m: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    tp = (m & y).sum()
    prec = tp / m.sum() if m.sum() else 0.0
    rec = tp / y.sum() if y.sum() else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec


def best_rule(X: pd.DataFrame, y: np.ndarray, atoms: list, max_terms: int = 3, beam: int = 20, min_gain: float = 0.01):
    """Лучшая конъюнкция интервальных условий для одного класса (лучевой поиск по F1).

    Условие — «признак ≥ порог» или «признак ≤ порог»; один признак может войти дважды с разными знаками
    (интервал [a, b]). Более длинное правило берется, только если F1 растет хотя бы на min_gain.
    """
    cand = [((a,), _f1(_mask(X, (a,)), y)[0]) for a in atoms]
    cand.sort(key=lambda z: -z[1])
    best = cand[0]
    frontier = cand[:beam]
    for _ in range(max_terms - 1):
        nxt = {}
        for rule, _ in frontier:
            used = {(f, op) for f, op, _ in rule}
            for a in atoms:
                if (a[0], a[1]) in used:
                    continue
                r = tuple(sorted(rule + (a,)))
                if r not in nxt:
                    nxt[r] = _f1(_mask(X, r), y)[0]
        if not nxt:
            break
        frontier = sorted(nxt.items(), key=lambda z: -z[1])[:beam]
        if frontier[0][1] >= best[1] + min_gain:
            best = frontier[0]
        else:
            break
    return best[0]


def interval_rules(X: pd.DataFrame, labels: pd.Series, qs: list[float], steps: dict, max_terms: int = 3,
                   beam: int = 20, min_gain: float = 0.01, folds: int = 5, seed: int = 42) -> dict:
    """Описание каждого типа интервальным правилом (в духе узорных структур FCA) + проверка на отложенных МО.

    Возвращает для каждого типа: правило, точность, полноту и F1 на всех МО, долю типа (точность случайного
    правила) и F1 на отложенных фолдах (правило ищется на остальных МО).
    """
    atoms = _atoms(X, qs, steps)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(X))
    out = {}
    for k in sorted(labels.unique()):
        y = (labels == k).to_numpy()
        rule = best_rule(X, y, atoms, max_terms, beam, min_gain)
        f1, prec, rec = _f1(_mask(X, rule), y)
        cv = []
        for i in range(folds):
            tr, te = fold != i, fold == i
            r_cv = best_rule(X[tr], y[tr], _atoms(X[tr], qs, steps), max_terms, beam, min_gain)
            cv.append(_f1(_mask(X[te], r_cv), y[te])[0])
        out[k] = {"rule": [{"feature": f, "op": op, "threshold": t} for f, op, t in rule],
                  "precision": round(float(prec), 3), "recall": round(float(rec), 3), "f1": round(float(f1), 3),
                  "prevalence": round(float(y.mean()), 3), "f1_cv_mean": round(float(np.mean(cv)), 3),
                  "support": int(_mask(X, rule).sum())}
    return out


def ordinal_signs(D: np.ndarray, eps: float) -> np.ndarray:
    """Знаки всех попарных сравнений показателей (Алескеров–Мячин): строки — объекты, столбцы — пары (u < v).

    +1, если d_u − d_v > eps; −1, если < −eps; 0 — «равны» в пределах допуска.
    """
    iu, iv = np.triu_indices(D.shape[1], 1)
    diff = D[:, iu] - D[:, iv]
    return np.where(diff > eps, 1, np.where(diff < -eps, -1, 0))


def ordinal_patterns(dev: pd.DataFrame, labels: pd.Series, eps: float) -> dict:
    """Порядковые паттерны типов: знаки попарных сравнений отклонений от среднего (правило Миркина) у центра типа
    и у каждого объекта. Возвращает долю совпадающих знаков объекта с паттерном своего типа и долю объектов, чей паттерн
    ближе к своему типу, чем к любому другому; расстояния Хэмминга между паттернами типов."""
    types = sorted(labels.unique())
    cent = np.array([dev[labels == k].mean().to_numpy() for k in types])
    P = ordinal_signs(cent, eps)
    S = ordinal_signs(dev.to_numpy(), eps)
    match = (S[:, None, :] == P[None, :, :]).mean(2)          # объекты × типы
    own = np.array([types.index(k) for k in labels])
    own_match = match[np.arange(len(own)), own]
    best = match.argmax(1)
    ham = {f"{a}-{b}": int((P[i] != P[j]).sum()) for i, a in enumerate(types) for j, b in enumerate(types) if i < j}
    return {"n_pairs": int(P.shape[1]), "eps": eps,
            "own_match_mean": {k: round(float(own_match[own == i].mean()), 3) for i, k in enumerate(types)},
            "closest_is_own_share": {k: round(float((best[own == i] == i).mean()), 3) for i, k in enumerate(types)},
            "closest_is_own_share_all": round(float((best == own).mean()), 3),
            "type_pattern_hamming": ham,
            "type_patterns": {k: P[i].tolist() for i, k in enumerate(types)}}


def rule_stability(X: pd.DataFrame, labels: pd.Series, rules: dict, qs: list[float], steps: dict, max_terms: int,
                   beam: int, min_gain: float, B: int, seed: int) -> dict:
    """Устойчивость интервальных правил к выборке объектов (бутстрап, в духе устойчивости понятий Кузнецова).

    Для каждого типа правило заново ищется на бутстрап-выборке. Доля повторов с тем же набором условий (признак и
    направление) и медианный Жаккар охвата (какие МО правило покрывает на всех данных) с исходным правилом.
    """
    rng = np.random.default_rng(seed)
    out = {}
    full_masks = {k: _mask(X, tuple((t["feature"], t["op"], t["threshold"]) for t in r["rule"])) for k, r in rules.items()}
    struct = {k: {(t["feature"], t["op"]) for t in r["rule"]} for k, r in rules.items()}
    acc = {k: {"same": 0, "jac": []} for k in rules}
    for _ in range(B):
        i = rng.integers(0, len(X), len(X))
        Xb, lb = X.iloc[i].reset_index(drop=True), labels.iloc[i].reset_index(drop=True)
        atoms = _atoms(Xb, qs, steps)
        for k in rules:
            r = best_rule(Xb, (lb == k).to_numpy(), atoms, max_terms, beam, min_gain)
            acc[k]["same"] += {(f, op) for f, op, _ in r} == struct[k]
            m = _mask(X, r)
            inter, uni = (m & full_masks[k]).sum(), (m | full_masks[k]).sum()
            acc[k]["jac"].append(inter / uni if uni else 1.0)
    for k, a in acc.items():
        out[k] = {"same_structure_share": round(a["same"] / B, 3), "extent_jaccard_median": round(float(np.median(a["jac"])), 3),
                  "extent_jaccard_q10": round(float(np.quantile(a["jac"], 0.1)), 3)}
    return out
