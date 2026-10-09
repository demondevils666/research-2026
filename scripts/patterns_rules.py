"""Этап 4: порядковые паттерны типов (Алескеров–Мячин) и устойчивость интервальных правил (в духе Кузнецова).

1) Показатели МО — отклонения от среднего по стране по правилу Миркина: шесть долей категорий и траты на жителя
   (2024). Паттерн — знаки всех попарных сравнений отклонений (21 пара), равенство в пределах допуска eps.
   Совпадает ли порядок «что выше среднего сильнее» у МО с паттерном его типа; различаются ли паттерны типов.
2) Устойчивость правил типов (types.json, E3): правило заново ищется на бутстрап-выборках МО.
Выход: outputs/stage3/patterns_rules.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, interpret  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    pr, rc = cfg["patterns"], cfg["rules"]
    proc, out = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    code = lab.map(interpret.type_names(lab)["code"])
    total = cfg["panel"]["total_category"]
    w = pd.read_parquet(proc / "panel_wide.parquet")
    wy = w[w.index.get_level_values("date").str.startswith(cfg["features"]["profile_year"])].groupby(level="territory_id").mean().loc[ids]
    X = pd.concat([features.shares(wy) * 100, wy[[total]]], axis=1)
    dev = (X - X.mean()) / X.mean().abs()
    res = {"definition": __doc__.split("Выход")[0].strip()}
    res["ordinal_patterns"] = interpret.ordinal_patterns(dev, code, pr["eps"])
    ty = json.loads((out / "types.json").read_text())
    rules = {c: t["rule"] for c, t in ty["types"].items()}
    steps = {c: rc["round_steps"].get(c, rc["round_steps"]["default"]) for c in X.columns}
    res["rule_stability"] = interpret.rule_stability(X, code, rules, rc["quantiles"], steps, rc["max_terms"], rc["beam"],
                                                     rc["min_gain"], pr["rule_bootstrap"], cfg["seed"])
    (out / "patterns_rules.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
