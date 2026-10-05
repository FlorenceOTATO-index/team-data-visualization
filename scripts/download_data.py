"""Download the raw dataset into data/raw/.

Fill in DATA_URL for our chosen dataset, then run:
    python scripts/download_data.py
"""
from pathlib import Path
import urllib.request

DATA_URL = ""  # TODO: dataset download URL
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_URL:
        raise SystemExit("Set DATA_URL in scripts/download_data.py first.")
    dest = RAW_DIR / DATA_URL.rsplit("/", 1)[-1]
    print(f"Downloading {DATA_URL} -> {dest}")
    urllib.request.urlretrieve(DATA_URL, dest)
    print("Done.")


if __name__ == "__main__":
    main()
