"""Этап 3b, E4: онлайн в аграрных районах (T6) — замещение офлайн-торговли или артефакт безналичного измерения?

Сельский тип с самыми низкими тратами (аграрные районы T6) выделяется высокой долей маркетплейсов в безналичных тратах. Два объяснения:
  замещение — где мало магазинов, люди покупают онлайн (тогда доля маркетплейсов падает с плотностью торговли);
  артефакт  — в селе офлайн-покупки чаще за наличные, их не видно, и доля онлайна в безнале завышена
              (тогда доля маркетплейсов растет там, где безнала мало относительно доходов).
Данные Росстата (БД ПМО, 2023), в кластеризацию не входили:
  площадь торговых залов (магазины + павильоны, IV квартал) на 1 000 жителей — плотность офлайн-торговли;
  оборот розничной торговли без малого бизнеса (ОКВЭД2 47, январь–декабрь) на жителя — присутствие крупных сетей;
  проникновение безнала — траты по картам на жителя (СберИндекс, 2023) / средняя зарплата (Росстат, место работы).
Проверки: медианы по типам; ранговые корреляции с долей маркетплейсов (все МО и сельские типы — configs/types.yaml: rural);
регрессия доли маркетплейсов на стандартизованные логарифмы плотности торговли и проникновения безнала
(плюс уровень зарплаты), бутстрап-ДИ коэффициентов. Правило: замещение подтверждено, если коэффициент плотности
торговли отрицателен и его 95%-ДИ не содержит 0 при контроле на проникновение безнала.
Выход: outputs/stage3/cash_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import data, interpret, rosstat  # noqa: E402
from mo.config import load, path  # noqa: E402


def retail(b: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rc = cfg["retail"]
    year = cfg["rosstat"]["year"]
    omap = rosstat.oktmo_map(b, year)
    raw = path("raw") / "rosstat" / "indicators"
    a = pd.read_csv(raw / f"data_{rc['area']}_year{year}_112_v20250918.csv", sep=";", dtype=str)
    a = rosstat._attach(a, omap)
    a = a[a["obroz"].isin(rc["area_objects"])]
    a["prio"] = (a["indicator_period"] != rc["area_period"]).astype(int)
    a = a[a["prio"] == a.groupby("territory_id")["prio"].transform("min")]
    area = a.drop_duplicates(["territory_id", "oktmo", "obroz"]).groupby("territory_id")["value"].sum()
    t = pd.read_csv(raw / f"data_{rc['turnover']}_112_v20250918.csv", sep=";", dtype=str)
    t = t[t["year"] == str(year)]
    t = rosstat._attach(t, omap)
    t = t[(t["indicator_period"] == rc["turnover_period"]) & (t["okved2"] == rc["turnover_okved"])]
    turn = t.drop_duplicates(["territory_id", "oktmo"]).groupby("territory_id")["value"].sum()
    return pd.DataFrame({"area_m2": area, "turnover_thr": turn})


def ols_boot(df: pd.DataFrame, y: str, xs: list[str], n: int, seed: int) -> dict:
    d = df[[y] + xs].dropna()
    Z = (d - d.mean()) / d.std(ddof=0)
    X = np.column_stack([np.ones(len(Z)), Z[xs].to_numpy()])
    Y = Z[y].to_numpy()
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(Z), len(Z))
        bs.append(np.linalg.lstsq(X[i], Y[i], rcond=None)[0])
    bs = np.array(bs)
    r2 = 1 - ((Y - X @ beta) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return {"n": int(len(d)), "r2": round(float(r2), 3),
            "beta_std": {x: {"b": round(float(beta[k + 1]), 3),
                             "ci95": [round(float(np.quantile(bs[:, k + 1], q)), 3) for q in (0.025, 0.975)]}
                         for k, x in enumerate(xs)}}


def main() -> None:
    cfg = load()
    seed, nb = cfg["seed"], cfg["probes"]["bootstrap"]
    proc, out = path("processed"), path("outputs") / "stage3"
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    code = lab.map(interpret.type_names(lab)["code"])
    b = data.borders()
    r = retail(b, cfg).reindex(ids)

    w = pd.read_parquet(proc / "panel_wide.parquet")
    pc = cfg["panel"]
    dates = w.index.get_level_values("date")
    y_rs = str(cfg["rosstat"]["year"])
    card = w[dates.str.startswith(y_rs)][pc["total_category"]].groupby(level="territory_id").mean()
    wp = w[dates.str.startswith(cfg["features"]["profile_year"])].groupby(level="territory_id").mean()
    mp = wp[cfg["retail"]["marketplace_category"]] / wp[pc["total_category"]]
    food = wp[cfg["retail"]["food_category"]] / wp[pc["total_category"]]

    df = pd.DataFrame({"type": code, "mp_share": mp, "food_share": food, "pop": mo["population"], "wage": mo["wage"],
                       "card_pc": card}).join(r)
    df["area_per_1000"] = df["area_m2"] / df["pop"] * 1000
    df["turnover_pc_month"] = df["turnover_thr"] * 1000 / df["pop"] / 12
    df["card_to_wage"] = df["card_pc"] / df["wage"]
    df["card_to_turnover"] = df["card_pc"] / df["turnover_pc_month"]
    for c in ("area_per_1000", "turnover_pc_month", "card_to_wage", "wage"):
        df[f"log_{c}"] = np.log(df[c].where(df[c] > 0))

    res = {"definition": __doc__.split("Выход")[0].strip()}
    res["coverage"] = {c: round(float(df[c].notna().mean()), 3) for c in ("area_per_1000", "turnover_pc_month", "card_to_wage")}
    med = df.groupby("type")[["mp_share", "food_share", "area_per_1000", "turnover_pc_month", "card_to_wage",
                              "card_to_turnover", "card_pc", "wage"]].median()
    res["type_medians"] = {t: {k: round(float(v), 4) for k, v in row.items()} for t, row in med.iterrows()}

    def spear(sub: pd.DataFrame) -> dict:
        outd = {}
        for x in ("area_per_1000", "turnover_pc_month", "card_to_wage", "wage"):
            d = sub[["mp_share", x]].dropna()
            rho, p = spearmanr(d["mp_share"], d[x])
            outd[x] = {"rho": round(float(rho), 3), "p": float(p), "n": int(len(d))}
        return outd

    rural = df["type"].isin(interpret.rural_codes())
    res["rural_types"] = interpret.rural_codes()
    res["spearman_mp_share"] = {"all": spear(df), "rural": spear(df[rural])}
    xs = ["log_area_per_1000", "log_card_to_wage", "log_wage"]
    res["regression_mp_share"] = {"all": ols_boot(df, "mp_share", xs, nb, seed),
                                  "rural": ols_boot(df[rural], "mp_share", xs, nb, seed),
                                  "rural_with_chains": ols_boot(df[rural], "mp_share", xs + ["log_turnover_pc_month"], nb, seed)}
    reg = res["regression_mp_share"]["rural"]["beta_std"]
    area_ci = reg["log_area_per_1000"]["ci95"]
    cash_ci = reg["log_card_to_wage"]["ci95"]
    res["verdict"] = {
        "substitution_supported": bool(area_ci[1] < 0),
        "artifact_signal": bool(cash_ci[1] < 0),
        "rule": "замещение: коэффициент плотности торговли < 0 и ДИ без нуля (сельские типы, контроль на безнал и зарплату); "
                "сигнал артефакта: коэффициент проникновения безнала < 0 и ДИ без нуля",
    }
    (out / "cash_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, ensure_ascii=False, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
