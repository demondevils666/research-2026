"""Скачивает код авторов (KEFRiN, CANUS — Шалилех; Pattern — лаборатория жюри) в third_party/ на закрепленных коммитах.

Код в репозиторий не копируется (у KEFRiN и CANUS нет файла LICENSE, Pattern под GPL-3.0); third_party/ в .gitignore.
Используется только для сверки нашей реализации KEFRiN и индексов (тесты) и для запуска CANUS.
Запуск: python scripts/fetch_external.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mo.config import load  # noqa: E402


def main() -> None:
    tp = ROOT / "third_party"
    tp.mkdir(exist_ok=True)
    for name, spec in load()["external"].items():
        d = tp / name
        if not d.exists():
            subprocess.run(["git", "clone", "-q", spec["repo"], str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", spec["commit"]], check=False)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", spec["commit"]], check=True)
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        print(f"{name}: {head}")


if __name__ == "__main__":
    main()
