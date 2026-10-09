"""Перепись 2010 (ВПН-2010), региональные издания «Том 7», таблица «Занятое в экономике население… по территории
нахождения работы и статусу по городским округам и муниципальным районам».

В xlsx и в тексте PDF (pdftotext -layout) одна структура: строка с названием МО, затем блоки «Городское и сельское
население» / «Городское население» / «Сельское население» с числом занятых и строками «на территории своего субъекта»,
«из них на территории своего населенного пункта», «на территории другого субъекта», «на территории других стран»,
«без указания территории нахождения работы». Берется первый числовой столбец (все занятые).
Мера: доля работающих в другом населенном пункте своего региона = (свой субъект − свой НП) / (занятые − без указания).
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

BLOCKS = ("городское и сельское население", "городское население", "сельское население")
UNIT = re.compile(r"район|округ|^г\.|^город|^зато|^закрытое|муниципальн", re.I)
NOT_UNIT = re.compile(r"поселени|сельсовет|внутригородск|администрат|насел[её]нн|занят|статус|экономическ|территори|"
                      r"работа|по найму|итоги|таблиц|продолжение|районам|^городское|^сельское|^все население|"
                      r"област|\bкрай\b|республик|автономн", re.I)                 # итог по региону — не МО


def _num(v) -> float | None:
    s = str(v).replace("\xa0", "").replace(" ", "").strip()
    if s in ("-", "–", "—"):
        return 0.0
    if s in ("K", "К", "...", "…"):          # данные скрыты (конфиденциальность): место в строке сохраняем
        return float("nan")
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def lines_xlsx(path: Path, sheet: str | None = None) -> list[tuple[str, list[float]]]:
    """Строки таблицы как (текст метки, числа справа от нее)."""
    if path.suffix == ".xls":
        import xlrd
        ws = xlrd.open_workbook(str(path)).sheet_by_name(sheet) if sheet else xlrd.open_workbook(str(path)).sheet_by_index(0)
        rows = (ws.row_values(i) for i in range(ws.nrows))
    else:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        rows = (wb[sheet] if sheet else wb.worksheets[0]).iter_rows(values_only=True)
    return [r for r in (_row(r) for r in rows) if r]


def _row(r) -> tuple[str, list[float]] | None:
    """Ячейки строки таблицы → (первая ячейка с буквами, числа правее нее)."""
    cells = [c for c in r if c is not None and str(c).strip() != ""]
    k = next((i for i, c in enumerate(cells) if re.search(r"[А-Яа-яЁё]", str(c))), None)
    if k is None:
        return None
    nums = [x for x in (_num(c) for c in cells[k + 1:]) if x is not None]
    return " ".join(str(cells[k]).split()), nums


def lines_html(path: Path, start: str, end: str | None = None, encoding: str = "cp1251") -> list[tuple[str, list[float]]]:
    """Таблица из документа Word, сохраненного как HTML: строки <tr> между абзацами-заголовками start и end."""
    from lxml import html as lh
    doc = lh.fromstring(path.read_bytes().decode(encoding, errors="replace"))
    out, on = [], False
    for el in doc.iter("p", "tr"):
        if el.tag == "tr":
            r = _row([" ".join(td.text_content().split()) for td in el.findall("td")]) if on else None
            if r:
                out.append(r)
            continue
        txt = " ".join(el.text_content().split())
        if not on and txt.startswith(start):
            on = True
        elif on and end and txt.startswith(end):
            break
    if not on:
        raise ValueError(f"{path.name}: нет заголовка таблицы «{start}»")
    return out


def lines_pdf(path: Path, start: str, end: str | None = None) -> list[tuple[str, list[float]]]:
    """Текст PDF (pdftotext -layout) от заголовка таблицы start до заголовка end: строки (текст, числа в конце)."""
    import subprocess
    text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout
    i = text.find(start)
    if i < 0:
        raise ValueError(f"{path.name}: нет заголовка таблицы «{start}»")
    j = text.find(end, i + len(start)) if end else -1
    out = []
    for line in text[i:j if j > 0 else None].split("\n"):
        m = re.match(r"^\s*(.*?[А-Яа-яЁё.)»\"])\s*((?:\s+(?:\d+|-|–|—|K|К|\.\.\.|…))*)\s*$", line)
        if not m or not m.group(1).strip():
            continue
        nums = [x for x in (_num(t) for t in m.group(2).split()) if x is not None]
        out.append((" ".join(m.group(1).split()), nums))
    return out


def _block(text: str, start: bool = False) -> str | None:
    """Вид блока по метке строки; start=True — и для первой части метки, разбитой на две строки («Городское и
    сельское» / «население»). «Все население», «Все (сельское) население», «Все население — городское» — итог по МО."""
    if re.match(r"все\b.*населени", text):
        return BLOCKS[0]
    if start and text.startswith("городское и сельское"):
        return BLOCKS[0]
    return next((b for b in BLOCKS if text.startswith(b)), None)


def parse(lines: list[tuple[str, list[float]]]) -> pd.DataFrame:
    """Записи по МО: занятые, свой субъект, свой НП, другой субъект, другие страны, без указания (блок «городское и
    сельское» или «все население», иначе первый блок).
    Метки в PDF бывают разбиты на две-три строки («из них на территории» / «своего населенного» / «пункта 1234»):
    строки без чисел копятся и приклеиваются к строке с числами; каждое поле блока заполняется один раз.
    Строка без чисел — название МО, если в ней есть вид МО (район, округ, «г.») или за ней сразу начинается блок
    («Великий Новгород» / «Все население 1234»). Название на двух строках склеивается, если во второй только вид МО."""
    # строка-название, у которой в PDF справа один прочерк («Арсеньевский городской округ -»), — тоже без чисел;
    # обрывки шапки, слипшиеся с названием («ников Городской округ»), отбрасываются
    lines = [(re.sub(r"^(?:(?:ни)?ков\s+)+(?=[А-ЯЁ\"«])", "", label), [] if len(nums) == 1 and nums[0] == 0 else nums)
             for label, nums in lines]
    lows = [label.lower().replace("ё", "е").rstrip(":") for label, _ in lines]
    units, unit, block, buf, prev = {}, None, None, [], ""
    for i, (label, nums) in enumerate(lines):
        low = lows[i]
        if not nums:
            nxt = lows[i + 1] if i + 1 < len(lines) else ""
            looks = UNIT.search(label) or (_block(nxt, start=True) and re.match(r"[А-ЯЁ]", label) and not label.endswith("-"))
            name_ok = not NOT_UNIT.search(label) and len(label) < 90
            dangling = unit is not None and block is None and (not norm_name(unit) or re.search(r"(–|—|\bг\.|«|\")$", unit))
            if dangling and name_ok and re.match(r"[А-ЯЁA-Z\"«]", label):
                # продолжение названия: «Городской округ» / «"Город Киров"», «Кемеровский городской округ – г.» / «Кемерово»
                unit = f"{unit} {label}"
            elif looks and name_ok:
                # «Бежаницкий муниципальный» / «район», «Новосокольнический» / «муниципальный район»
                unit = f"{prev} {label}" if prev and not norm_name(label) else label
                block, buf = None, []
            else:
                buf.append(low)
            prev = label
            continue
        prev = ""
        kind = _block(low) or (_block(f"{buf[-1]} {low}") if buf else None)
        full = " ".join(buf[-2:] + [low])
        buf = []
        if unit is None:
            continue
        if kind:
            block = {"unit": unit, "block": kind, "employed": nums[0]}
            units.setdefault(unit, []).append(block)
            continue
        if block is None or nums[0] != nums[0]:            # NaN: значение скрыто
            continue
        field = ("own_region" if "своего" in full and "субъект" in full else
                 "own_settlement" if "своего" in full and "населен" in full else
                 "other_region" if "другого" in full and "субъект" in full else
                 "abroad" if "других стран" in full or "зарубежных стран" in full else
                 "unknown" if "без указания" in full or "нахождения работы" in full else None)
        if field and field not in block:
            block[field] = nums[0]
    rows = []
    for u, blocks in units.items():
        b = next((x for x in blocks if x["block"] == BLOCKS[0]), blocks[0])
        if {"own_region", "own_settlement"} <= set(b):
            rows.append(b)
    return _share(pd.DataFrame(rows))


def parse_wide(lines: list[tuple[str, list[float]]], columns: list[str]) -> pd.DataFrame:
    """Таблица «одна строка на МО» (Башкортостан): числа строки по порядку columns; строки «на 1000 человек» и
    подписи без полного набора чисел пропускаются. Если есть столбец known (указавшие территорию), без указания =
    employed − known."""
    rows = []
    for label, nums in lines:
        if len(nums) < len(columns) or "на 1000" in label.lower():
            continue
        r = {"unit": label, "block": BLOCKS[0], **dict(zip(columns, nums))}
        if "known" in r:
            r["unknown"] = r["employed"] - r.pop("known")
        rows.append(r)
    return _share(pd.DataFrame(rows))


def _share(df: pd.DataFrame) -> pd.DataFrame:
    """share_other_settlement — работают в другом НП своего региона; share_outside_settlement — вне своего НП вообще
    (другой НП своего региона, другой регион, другие страны). Знаменатель — занятые, указавшие территорию работы."""
    if len(df):
        unknown = df["unknown"].fillna(0) if "unknown" in df else 0
        known = df["employed"] - unknown
        df["share_other_settlement"] = (df["own_region"] - df["own_settlement"]) / known
        df["share_outside_settlement"] = (known - df["own_settlement"]) / known
    return df


def read_region(raw: Path, spec: dict) -> pd.DataFrame:
    """Таблица одного региона по описанию из configs/base.yaml: файл, лист, запись в zip или заголовки таблицы в PDF."""
    path = raw / spec["file"]
    if spec.get("member"):                     # файл внутри распакованного архива: ищем по шаблону имени
        found = sorted(path.parent.joinpath("extracted").rglob(spec["member"]))
        if not found:
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if Path(n).match(spec["member"])]
            raise FileNotFoundError(f"{path.name}: нет {spec['member']} среди распакованных (в архиве: {names[:3]})")
        path = found[0]
    if path.suffix == ".pdf":
        lines = lines_pdf(path, spec["start"], spec.get("end"))
    elif path.suffix in (".htm", ".html"):
        lines = lines_html(path, spec["start"], spec.get("end"))
    else:
        lines = lines_xlsx(path, spec.get("sheet"))
    return parse_wide(lines, spec["columns"]) if spec.get("layout") == "wide" else parse(lines)


def norm_name(name: str) -> str:
    """Ключ для сопоставления названий МО 2010 года и справочника: без видов МО, кавычек и скобок.
    «Городской округ г. Кызыл – г. Кызыл»: после тире с пробелами — название центра, оно отбрасывается."""
    s = name.lower().replace("ё", "е")
    s = re.split(r"\s[–—-]\s", s)[0]
    s = re.sub(r"\(.*?\)|[«»\"]", " ", s)
    s = re.sub(r"(^|\s)(г|им|пгт|п|рп)\.", " ", s)
    s = re.sub(r"\b(муниципальное образование|муниципальный|муниципального|район[а]?|округ[а]?|городской|городское|"
               r"город|поселение|поселок|пгт|рп|имени|кожуун|закрытое административно-территориальное образование|"
               r"зато)\b", " ", s)
    s = re.sub(r"[^а-я0-9 -]", " ", s)
    s = re.sub(r"(\w)(ское|ского|ская|ской)\b", r"\1ский", s)     # «Усольское МО» = «Усольский район»
    return " ".join(s.split())


def kind(name: str) -> str:
    """Вид МО по названию: «district» (район, муниципальный округ, кожуун), «city» (городской округ, город) или «?»."""
    s = name.lower()
    if not s.startswith("городской округ") and re.search(r"муниципальн\w* (район|округ|образование)|район|кожуун", s):
        return "district"
    if re.search(r"городской округ|^г\.|\bгород\b|зато", s):
        return "city"
    return "?"


def match(df: pd.DataFrame, directory: pd.Series, aliases: dict | None = None) -> tuple[pd.DataFrame, list, list]:
    """Записи переписи одного региона → territory_id справочника (directory: territory_id → название) по norm_name.
    Если ключ в справочнике не один («Камышловский городской округ» и «Камышловский муниципальный район»), выбирается
    МО того же вида. Город с прилагательным в названии («Рубцовский городской округ») ищется и как «Рубцовск».
    aliases — ручные соответствия ключей «перепись → справочник» (город переименован или назван иначе).
    Из записей, попавших на одно МО, берется первая: строка МО идет раньше строк его поселений («Городской округ
    город Ачинск», затем «г. Ачинск»). Возвращает таблицу с territory_id, неопознанные записи переписи и МО без пары."""
    aliases = aliases or {}
    keys = {}
    for tid, name in directory.items():
        keys.setdefault(norm_name(name), []).append((tid, kind(name)))

    def find(unit: str, strict: bool):
        k, kd = norm_name(unit), kind(unit)
        tries = [aliases.get(k, k)]
        if kd == "city" and k.endswith("ский"):
            tries.append(k[:-4] + "ск")
        for key in tries:
            cand = keys.get(key, [])
            if len(cand) > 1 or (strict and kd != "?"):
                cand = [c for c in cand if c[1] == kd or c[1] == "?"]
            if len(cand) == 1:
                return cand[0][0]
        return None

    # Сначала совпадение вида МО («Бийский городской округ» — не «Бийский район»), затем для оставшихся МО — без
    # него: часть районов 2010 года потом стала городскими или муниципальными округами («Гусевский район»)
    df = df.copy()
    df["key"] = [norm_name(u) for u in df["unit"]]
    df["territory_id"] = [find(u, True) for u in df["unit"]]
    taken = set(df["territory_id"].dropna())
    for i in df.index[df["territory_id"].isna()]:
        t = find(df.at[i, "unit"], False)
        if t is not None and t not in taken:
            df.at[i, "territory_id"] = t
            taken.add(t)
    hit = df[df["territory_id"].notna()].drop_duplicates("territory_id")
    unmatched = df.loc[df["territory_id"].isna(), "unit"].tolist()
    missing = [directory[t] for t in directory.index if t not in set(hit["territory_id"])]
    return hit.astype({"territory_id": int}), unmatched, missing
