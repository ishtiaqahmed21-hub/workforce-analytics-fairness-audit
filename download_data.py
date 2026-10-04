"""Download exact public sources, validating hashes before saving."""

from pathlib import Path
from urllib.request import urlopen
import hashlib

SOURCES = {
    "emp_attrition.csv": (
        "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv",
        "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659",
    ),
    "adult.zip": (
        "https://archive.ics.uci.edu/static/public/2/adult.zip",
        "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb",
    ),
}


def download():
    destination = Path(__file__).resolve().parent / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        target = destination / name
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        ):
            print(f"Verified cached {name}")
            continue
        with urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"{name} hash mismatch; investigate upstream change")
        target.write_bytes(content)
        print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    download()
