"""Чтение zip-архива по HTTP без скачивания целиком (через Range-запросы).

Разделы БД ПМО Росстата весят до 3,6 ГБ, а нам нужны отдельные показатели.
zipfile читает оглавление в конце файла и затем только нужные записи,
поэтому достаточно дать ему «файл», который подкачивает куски по запросу.
"""

import io
import time

import requests


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, verify: str, chunk: int = 1 << 20, tries: int = 6):
        self.url, self.verify, self.chunk, self.tries = url, verify, chunk, tries
        self.session = requests.Session()
        for k in range(tries):
            try:
                head = self.session.head(url, verify=verify, timeout=60, allow_redirects=True)
                head.raise_for_status()
                break
            except requests.RequestException:
                if k == tries - 1:
                    raise
                time.sleep(2 ** k)
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            a = i * self.chunk
            b = min(a + self.chunk, self.size) - 1
            for k in range(self.tries):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={a}-{b}"},
                                         verify=self.verify, timeout=120)
                    r.raise_for_status()
                    if len(r.content) != b - a + 1:
                        raise IOError(f"короткий ответ {len(r.content)} вместо {b - a + 1}")
                    self.cache[i] = r.content
                    break
                except Exception:
                    if k == self.tries - 1:
                        raise
                    time.sleep(2 ** k)
            if len(self.cache) > 256:  # держим в памяти не больше ~256 МБ
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.chunk)
            piece = self._block(i)[off:off + n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)
