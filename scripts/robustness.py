"""Этап 3, шаг 4: надежность итоговой типологии (KEFRiN, физическая сеть, K и α из этапа 2).

Сравниваем с итоговой типологией (ARI) варианты:
  другие seed (5 штук); годовой профиль 2023 вместо 2024; без внутригородских территорий Москвы и СПб;
  только структура трат (без уровня); дорожная сеть kNN вместо границ; K = 5 (вложенность типов, кросс-таблица).
Потолок ARI с регионами: чистая и случайные регионализации на K групп.
Выход: outputs/stage3/robustness.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mo import features, interpret, methods, networks, panel  # noqa: E402
from mo.config import load, path  # noqa: E402


def main() -> None:
    cfg = load()
    seed, n_init = cfg["seed"], cfg["methods"]["n_init"]
    proc, out = path("processed"), path("outputs") / "stage3"
    K = json.loads((path("outputs") / "stage2" / "select_k.json").read_text())["chosen_K"]
    alpha = json.loads((path("outputs") / "stage2" / "kefrin_sweep.json").read_text())["chosen_alpha"]
    mo = pd.read_parquet(proc / "mo.parquet")
    ids = sorted(mo.index)
    lab = pd.read_csv(path("outputs") / "stage2" / "labels_static.csv", index_col="territory_id")["type"].loc[ids]
    code_of = interpret.type_names(lab)["code"].to_dict()
    prof = pd.read_parquet(proc / "profile.parquet").loc[ids]
    Y = features.standardize(prof).to_numpy()
    W = networks.to_dense(pd.read_parquet(proc / "net_physical.parquet"), ids)
    base = lab.to_numpy()
    res = {"K": K, "alpha": alpha, "variants": {}}

    def ari(x, mask=None):
        return round(float(adjusted_rand_score(base if mask is None else base[mask], x)), 3)

    res["variants"]["seeds"] = [ari(methods.kefrin(Y, W, K, alpha, s, n_init)[0]) for s in (1, 7, 13, 21, 99)]

    w = panel.wide()
    Y23 = features.standardize(features.profile(w, "2023").loc[ids]).to_numpy()
    l23 = methods.align_labels(base, methods.kefrin(Y23, W, K, alpha, seed, n_init)[0])
    res["variants"]["profile_2023"] = ari(l23)
    # Макроуровень: сельские и малогородские типы (configs/types.yaml: rural) объединены — устойчива ли крупная структура
    rural = {k for k, c in code_of.items() if c in interpret.rural_codes()}
    macro = lambda x: np.array([-1 if v in rural else v for v in x])  # noqa: E731
    res["rural_types"] = interpret.rural_codes()
    res["variants"]["profile_2023_macro_rural_merged"] = round(float(adjusted_rand_score(macro(base), macro(l23))), 3)
    res["variants"]["profile_2023_same_type_share"] = round(float((l23 == base).mean()), 3)

    keep = ~mo.loc[ids, "type"].str.startswith("внутригородская").to_numpy()
    sub = methods.kefrin(features.standardize(prof[keep]).to_numpy(), W[np.ix_(keep, keep)], K, alpha, seed, n_init)[0]
    res["variants"]["without_moscow_spb_districts"] = ari(sub, keep)

    clr_cols = [c for c in prof.columns if c.startswith("clr_")]
    res["variants"]["structure_only"] = ari(methods.kefrin(features.standardize(prof[clr_cols]).to_numpy(), W, K, alpha, seed, n_init)[0])

    Wr = networks.to_dense(pd.read_parquet(proc / "net_road.parquet"), ids)
    res["variants"]["road_knn_network"] = ari(methods.kefrin(Y, Wr, K, alpha, seed, n_init)[0])

    # Соседние K: насколько разбиение меняется при K − 1 и K + 1 и как итоговые типы делятся или сливаются
    res["variants"]["k_alt_ari"], res["k_alt_crosstab"] = {}, {}
    for kk in (K - 1, K + 1):
        lk = methods.kefrin(Y, W, kk, alpha, seed, n_init)[0]
        ct = pd.crosstab(pd.Series(base).map(code_of).to_numpy(), lk)
        res["variants"]["k_alt_ari"][str(kk)] = ari(lk)
        res["k_alt_crosstab"][str(kk)] = {r: {f"K{kk}_{c}": int(v) for c, v in row.items() if v} for r, row in ct.iterrows()}
    res["k_alt_note"] = "строки — итоговые типы, столбцы — кластеры при соседнем K"
    # Второй согласованный вариант (сначала α, потом K): насколько он похож на итоговую типологию и как ложатся типы
    ap = cfg["robustness"]["alt_pair"]
    la = methods.kefrin(Y, W, ap["K"], ap["alpha"], seed, n_init)[0]
    ct = pd.crosstab(pd.Series(base).map(code_of).to_numpy(), la)
    res["alt_pair"] = {"K": ap["K"], "alpha": ap["alpha"], "ari": ari(la),
                       "crosstab": {r: {f"c{c}": int(v) for c, v in row.items() if v} for r, row in ct.iterrows()}}
    # Потолок ARI с регионами: насколько похожим на регионы может быть разбиение на K групп. Чистая регионализация —
    # каждый регион целиком в типе большинства своих МО; случайная — регионы случайно разложены по K группам.
    # ARI типологии с регионами сравниваем с этим потолком, а не с нулем (при K, много меньшем числа регионов, он низок)
    region = mo.loc[ids, "region"].to_numpy()
    maj = pd.Series(base).groupby(region).agg(lambda x: x.mode().iloc[0])
    pure = pd.Series(region).map(maj).to_numpy()
    regs = np.unique(region)
    rng = np.random.default_rng(seed)
    rand = [adjusted_rand_score(region, pd.Series(region).map(dict(zip(regs, rng.integers(0, K, len(regs))))).to_numpy())
            for _ in range(cfg["robustness"]["region_ceiling_draws"])]
    res["region_ari_ceiling"] = {
        "typology_ari": round(float(adjusted_rand_score(region, base)), 3),
        "pure_regionalization_ari": round(float(adjusted_rand_score(region, pure)), 3),
        "random_regionalization_ari_median": round(float(np.median(rand)), 3),
        "random_regionalization_ari_max": round(float(np.max(rand)), 3),
        "typology_nmi": round(float(normalized_mutual_info_score(region, base)), 3),
        "pure_regionalization_nmi": round(float(normalized_mutual_info_score(region, pure)), 3),
        "n_regions": int(len(regs)), "n_random": len(rand)}
    (out / "robustness.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
