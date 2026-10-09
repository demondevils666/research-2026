"""Собирает CA-бандл для скачивания данных с сайтов СберИндекса и Росстата.

Зачем: часть сайтов отдает неполную цепочку сертификатов (sberindex.ru не присылает
промежуточный LiteSSL RSA CA 2025), а www.sberbank.com и rosstat.gov.ru подписаны
корнем Минцифры (Russian Trusted Root CA), которого нет в стандартных хранилищах.
Проверку TLS мы не отключаем: добавляем недостающие сертификаты в отдельный бандл
и передаем его только нашим загрузкам.

Откуда сертификаты в certs/ (публичные, не секреты):
- russian_trusted_root_ca.pem: https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt
  (Госуслуги); отпечаток совпал с корнем, который отдает www.sberbank.com.
- russian_trusted_sub_ca_2024.pem: из цепочки www.sberbank.com, проверен
  `openssl verify` против корня выше.
- litessl_rsa_ca_2025.pem: из цепочки sber.ru, проверен против системного
  TrustAsia/Certum.

Запуск: python scripts/build_ca_bundle.py  ->  certs/bundle.pem (в git не кладется).
"""

import hashlib
import os
import ssl
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
CERTS = ROOT / "certs"
OUT = CERTS / "bundle.pem"

# SHA-256 отпечатки DER; если файл подменят, сборка упадет.
PINNED = {
    "russian_trusted_root_ca.pem": "D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31",
    "russian_trusted_sub_ca_2024.pem": "2155785036C900DBB5F1BB2A1569C80C55595BD6BF94867A29BBDDBC7D88A3F2",
    "litessl_rsa_ca_2025.pem": "AD4DF1416B387E6ABBF8612B5807C2A2B404F6AA3AA50EAB360FDE30EB6F0F11",
}


def fingerprint(pem: str) -> str:
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest().upper()


def base_bundle() -> Path:
    # В облачной сессии трафик идет через прокси со своим CA: берем его бандл.
    for var in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"):
        p = os.environ.get(var)
        if p and Path(p).is_file():
            return Path(p)
    return Path(certifi.where())


def main() -> None:
    parts = [base_bundle().read_text()]
    for name, expected in PINNED.items():
        pem = (CERTS / name).read_text()
        got = fingerprint(pem)
        if got != expected:
            raise SystemExit(f"{name}: отпечаток {got} != ожидаемый {expected}")
        parts.append(f"# {name}\n{pem}")
    OUT.write_text("\n".join(parts))
    print(f"OK: {OUT} (база: {base_bundle()})")


if __name__ == "__main__":
    main()
