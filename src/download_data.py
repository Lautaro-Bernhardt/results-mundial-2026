"""Download historical international results from martj42/international_results."""
import os
import sys
import requests

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
GOALSCORERS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/goalscorers.csv"
)
OUT_DIR = "data"


def download(url: str, dest: str) -> bool:
    print(f"Downloading {url} ...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        kb = len(r.content) / 1024
        lines = r.content.count(b"\n")
        print(f"  Saved {dest} ({kb:.0f} KB, {lines:,} lines)")
        return True
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok = download(RESULTS_URL, os.path.join(OUT_DIR, "historical_results.csv"))
    download(GOALSCORERS_URL, os.path.join(OUT_DIR, "goalscorers.csv"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
