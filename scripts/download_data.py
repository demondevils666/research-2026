"""Скачивает исходные данные по configs/data_sources.yaml в data/raw/.

Идемпотентно: уже скачанный файл с тем же размером не качается повторно.
data/raw/MANIFEST.json (url, размер, sha256) хранится в git: это версия данных, на которой получены результаты.
Каждый скачанный файл сверяется с ним по sha256. Если файл у источника изменился, скрипт останавливается
с кодом 1 и манифест не переписывает: результаты на новой версии данных могут не совпасть с отчетом.
Принять новую версию — флаг --update-manifest. Проверить уже скачанные файлы без сети — флаг --verify.
TLS проверяется всегда, сертификаты: certs/bundle.pem (scripts/build_ca_bundle.py).

Запуск: python scripts/download_data.py [--only hackathon,borders] [--skip-rosstat] [--verify] [--update-manifest]
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from httpzip import HttpRangeFile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CA = str(ROOT / "certs" / "bundle.pem")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; sberindex contest)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, tries: int = 6) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120, verify=CA, headers=HEADERS) as r:
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", 0))
                if dest.exists() and expected and dest.stat().st_size == expected:
                    print(f"  есть: {dest.relative_to(ROOT)}")
                    return
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                if expected and tmp.stat().st_size != expected:
                    raise IOError(f"размер {tmp.stat().st_size} != {expected}")
                tmp.replace(dest)
                print(f"  скачано: {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} МБ)")
                return
        except Exception as e:  # сайты иногда рвут соединение, повторяем
            if k == tries - 1:
                raise
            print(f"  повтор {k + 1} для {url}: {e}")
            time.sleep(2 ** k)


def extract(archive: Path) -> None:
    out = archive.parent / "extracted"
    if out.exists() and any(out.iterdir()):
        return
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        try:
            with zipfile.ZipFile(archive) as z:
                # имена без флага UTF-8 в архивах Росстата — в cp866 (по умолчанию zipfile читает их как cp437)
                legacy = any(not i.flag_bits & 0x800 and not i.filename.isascii() for i in z.infolist())
            with zipfile.ZipFile(archive, metadata_encoding="cp866" if legacy else None) as z:
                z.extractall(out)
            print(f"  распаковано в {out.relative_to(ROOT)}")
            return
        except zipfile.BadZipFile:              # имена в оглавлении и заголовках в разных кодировках (cp866/cp1251)
            shutil.rmtree(out)
            out.mkdir(parents=True)
    # rar или zip с «кривыми» именами: нужен bsdtar (libarchive-tools) или unar
    tool = shutil.which("bsdtar") or shutil.which("unar")
    if not tool:
        raise SystemExit("Для .rar нужен bsdtar (apt install libarchive-tools) или unar")
    if not tool.endswith("bsdtar"):
        subprocess.run([tool, "-o", str(out), str(archive)], check=True)
        print(f"  распаковано в {out.relative_to(ROOT)}")
        return
    env = {**os.environ, "LC_ALL": "C.UTF-8"}          # имена в архивах Росстата — кириллица (UTF-16 в RAR)
    subprocess.run([tool, "-xf", str(archive), "-C", str(out)], env=env, capture_output=True)
    # Файлы с именем длиннее 255 байт файловая система не принимает: пишем их под коротким именем
    # (начало имени + хэш), шаблоны member в configs/base.yaml смотрят на начало имени
    names = subprocess.run([tool, "-tf", str(archive)], env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    for name in names:
        if name.endswith("/"):
            continue
        parts = list(Path(name).parts)
        if len(parts[-1].encode()) > 200:
            stem, ext = Path(parts[-1]).stem, Path(parts[-1]).suffix
            parts[-1] = f"{stem[:60]}~{hashlib.sha256(name.encode()).hexdigest()[:8]}{ext}"
        dest = out.joinpath(*parts)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            r = subprocess.run([tool, "-xOf", str(archive), name], env=env, stdout=f, stderr=subprocess.PIPE)
        if r.returncode != 0:                       # например, RAR-фильтры: такой файл пропускаем (нужные — проверяет анализ)
            dest.unlink()
            print(f"  не распаковано: {name[:80]}")
    print(f"  распаковано в {out.relative_to(ROOT)}")


def rosstat(cfg: dict) -> list[dict]:
    """Оглавление разделов БД ПМО + выбранные показатели (только нужные записи zip)."""
    out_dir = RAW / "rosstat"
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / "entries.csv"
    rows = []
    zips = {}
    for s in cfg["sections"]:
        z = zipfile.ZipFile(HttpRangeFile(cfg["section_url"].format(section=s), CA, chunk=1 << 18))
        zips[s] = z
        rows += [{"section": s, "name": i.filename, "size": i.file_size} for i in z.infolist()]
    with open(listing, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "size"])
        w.writeheader()
        w.writerows(rows)
    print(f"  оглавление БД ПМО: {len(rows)} файлов -> {listing.relative_to(ROOT)}")

    # Название каждого показателя: первые строки его основного csv
    names = []
    for r in rows:
        n = r["name"]
        if "/" in n or not n.endswith(".csv"):
            continue
        with zips[r["section"]].open(n) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig")
            lines = [text.readline().rstrip("\r\n"), text.readline().rstrip("\r\n")]
        hdr = lines[0].split(";")
        vals = lines[1].split(";") if len(lines) > 1 else [""] * len(hdr)
        rec = dict(zip(hdr, vals))
        names.append({"section": r["section"], "indicator_code": rec.get("indicator_code", ""),
                      "indicator_section": rec.get("indicator_section", ""),
                      "indicator_name": rec.get("indicator_name", ""), "unit": rec.get("indicator_unit", ""),
                      "size": r["size"]})
    with open(out_dir / "indicators.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(names[0].keys()))
        w.writeheader()
        w.writerows(names)
    print(f"  справочник показателей: {len(names)} -> data/raw/rosstat/indicators.csv")

    got = []
    for s, codes in cfg["indicators"].items():
        for item in codes:
            # Элемент списка — код показателя (годы по умолчанию) или {code, years}
            code, years = (item, cfg["years"]) if isinstance(item, str) else (item["code"], item["years"])
            for y in years:
                n = f"data_{code}_parts/data_{code}_year{y}_112_v20250918.csv"
                if n not in zips[s].namelist():
                    n = f"data_{code}_112_v20250918.csv" if y == years[0] else None
                if not n:
                    continue
                dest = out_dir / "indicators" / Path(n).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    with zips[s].open(n) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                got.append({"url": cfg["section_url"].format(section=s) + "#" + n, "path": dest})
    return got


def verify(manifest: list[dict]) -> int:
    """Сверяет файлы data/raw с манифестом по sha256 без скачивания. Возвращает число расхождений."""
    bad = 0
    for m in manifest:
        f = ROOT / m["path"]
        if not f.exists():
            print(f"  нет файла: {m['path']}")
            bad += 1
        elif sha256(f) != m["sha256"]:
            print(f"  sha256 не совпадает: {m['path']}")
            bad += 1
    print(f"{'OK' if not bad else 'ОШИБКА'}: проверено {len(manifest)} файлов, расхождений {bad}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-rosstat", action="store_true")
    ap.add_argument("--verify", action="store_true", help="только сверить скачанные файлы с MANIFEST.json")
    ap.add_argument("--update-manifest", action="store_true", help="принять новую версию файлов, если sha256 изменился")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "data_sources.yaml"))
    only = set(filter(None, args.only.split(",")))
    path = RAW / "MANIFEST.json"
    old = json.loads(path.read_text()) if path.exists() else []
    if args.verify:
        sys.exit(1 if verify([m for m in old if not only or m["key"] in only]) else 0)
    known = {m["path"]: m["sha256"] for m in old}

    manifest = []
    for key, item in cfg["files"].items():
        if only and key not in only:
            continue
        print(f"[{key}]")
        dest = RAW / item["dest"]
        fetch(item["url"], dest)
        if item.get("extract"):
            extract(dest)
        manifest.append({"key": key, "url": item["url"], "path": str(dest.relative_to(ROOT)),
                         "bytes": dest.stat().st_size, "sha256": sha256(dest)})
    if not args.skip_rosstat and (not only or "rosstat" in only):
        print("[rosstat]")
        for g in rosstat(cfg["rosstat"]):
            manifest.append({"key": "rosstat", "url": g["url"], "path": str(g["path"].relative_to(ROOT)),
                             "bytes": g["path"].stat().st_size, "sha256": sha256(g["path"])})
    # Сверка с зафиксированной версией данных
    changed = [m["path"] for m in manifest if m["path"] in known and known[m["path"]] != m["sha256"]]
    if changed and not args.update_manifest:
        print("ОШИБКА: эти файлы отличаются от версии в data/raw/MANIFEST.json (sha256):")
        for c in changed:
            print(f"  {c}")
        print("Результаты на новой версии данных могут не совпасть с отчетом. Принять ее: --update-manifest.")
        sys.exit(1)
    n_same = sum(m["path"] in known for m in manifest)
    print(f"sha256: {n_same} файлов совпадают с MANIFEST.json, новых {len(manifest) - n_same}, изменилось {len(changed)}")
    # Сливаем с прежним манифестом, чтобы частичный запуск (--only) его не затирал
    merged = {m["path"]: m for m in old}
    merged.update({m["path"]: m for m in manifest})
    path.write_text(json.dumps(sorted(merged.values(), key=lambda m: m["path"]), ensure_ascii=False, indent=1))
    print(f"OK: {len(manifest)} файлов в этом запуске, всего в data/raw/MANIFEST.json: {len(merged)}")


if __name__ == "__main__":
    main()
