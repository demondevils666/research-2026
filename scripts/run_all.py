"""Полный прогон пайплайна одной командой (после scripts/download_data.py).

Шаги: код авторов (KEFRiN, CANUS, Pattern) -> инвентаризация данных -> панель -> пробы этапа 1 -> сети -> выбор K ->
развертка α и проверка K при итоговом α -> сравнение методов (настройка CANUS, сравнение, сверка с кодом авторов,
KEFRiN на всех шести сетях, панель ICVI, 10 seed и тесты) -> динамика ->
интерпретация (типы, агломерации, пространство, надежность) -> дополнения (функциональные города, сеть по месяцам, двойники,
паттерны и правила типов, торговля, НДФЛ) -> рисунки -> отчет (md, html, pdf) -> лендинг -> тесты.
Все параметры в configs/base.yaml, seed фиксирован.
Паспорт эффективности: время и пиковая память каждого шага -> outputs/run_timings.json.
Запуск: python scripts/run_all.py [--from STEP] [--no-tests]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("external", ["scripts/fetch_external.py"]),
    ("inventory", ["scripts/inventory.py"]),
    ("panel", ["scripts/build_panel.py"]),
    ("probes", ["scripts/probes/p1_ring_pairs.py", "scripts/probes/p2_ring_level.py", "scripts/probes/p3_belt_archipelago.py"]),
    ("networks", ["scripts/run_networks.py"]),
    ("select_k", ["scripts/select_k.py"]),
    ("sweep", ["scripts/run_kefrin_sweep.py"]),
    ("compare", ["scripts/canus_tuning.py", "scripts/run_compare.py", "scripts/verify_implementations.py",
                 "scripts/edge_rules.py", "scripts/icvi_panel.py", "scripts/methods_seeds.py"]),
    ("dynamics", ["scripts/run_dynamics.py"]),
    ("interpret", ["scripts/interpret_types.py", "scripts/check_k_alpha.py", "scripts/agglomerations.py", "scripts/space_dynamics.py",
                   "scripts/robustness.py"]),
    ("extras", ["scripts/functional_cities.py", "scripts/cash_check.py", "scripts/monthly_network.py", "scripts/twins.py",
                "scripts/patterns_rules.py", "scripts/fiscal_check.py", "scripts/commute_check.py", "scripts/lab_check.py",
                "scripts/assign_partial.py"]),
    ("figures", ["scripts/make_figures.py"]),
    ("report", ["scripts/build_report.py"]),
    ("landing", ["scripts/build_landing.py"]),
]


def run(script: str) -> dict:
    """Запускает шаг и возвращает время и пиковую память дочернего процесса (wait4 дает rusage именно этого процесса)."""
    t = time.time()
    proc = subprocess.Popen([sys.executable, str(ROOT / script)], cwd=ROOT, stdout=subprocess.DEVNULL)
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, script)
    return {"script": script, "seconds": round(time.time() - t, 1), "cpu_seconds": round(ru.ru_utime + ru.ru_stime, 1),
            "peak_rss_mb": round(ru.ru_maxrss / 1024, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[s for s, _ in STEPS])
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()
    names = [s for s, _ in STEPS]
    t0 = time.time()
    rows = []
    for name, scripts in STEPS[names.index(args.start):]:
        for s in scripts:
            print(f"== {name}: {s}", flush=True)
            r = run(s)
            rows.append({"step": name, **r})
            print(f"   ok, {r['seconds']:.0f} с, пик памяти {r['peak_rss_mb']:.0f} МБ", flush=True)
    if not args.no_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True, cwd=ROOT)
    total = round(time.time() - t0, 1)
    (ROOT / "outputs" / "run_timings.json").write_text(json.dumps({
        "from_step": args.start, "total_seconds": total, "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "platform": platform.platform(), "steps": rows}, ensure_ascii=False, indent=1))
    print(f"Готово за {total:.0f} с; время и память шагов: outputs/run_timings.json")


if __name__ == "__main__":
    main()
