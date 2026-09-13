"""METR-LA acquisition helper.

Keep third-party data under data/external/metr-la/. The canonical public record is
Zenodo record 5146275. This script intentionally does not embed credentials.
"""
from pathlib import Path

URL = "https://zenodo.org/records/5146275"
DEST = Path(__file__).resolve().parents[1] / "data/external/metr-la"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("METR-LA public record:", URL)
    print("Place downloaded files under:", DEST)


if __name__ == "__main__":
    main()
