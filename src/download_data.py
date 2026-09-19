"""Download the California Housing dataset.

Uses the Kaggle API if you have a kaggle.json saved. If you do not, it falls
back to a public copy of the exact same file, so this always works.
"""

import shutil
import subprocess
import urllib.request
from pathlib import Path

# Find the project folder: this file is in src/, so go up one level.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / "housing.csv"

KAGGLE_DATASET = "camnugent/california-housing-prices"
BACKUP_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/"
    "master/datasets/housing/housing.csv"
)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Already downloaded? Then there is nothing to do.
    if TARGET.exists():
        print("housing.csv is already here, skipping the download.")
        return

    # Do we have Kaggle login details saved?
    has_kaggle_login = (Path.home() / ".kaggle" / "kaggle.json").exists()

    if has_kaggle_login:
        print("Downloading from Kaggle...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(DATA_DIR), "--unzip"],
            check=True,
        )
    else:
        print("No Kaggle login found, using the public backup copy instead.")
        print("(To use Kaggle, save your kaggle.json into ~/.kaggle/)")
        with urllib.request.urlopen(BACKUP_URL) as web_file:
            with open(TARGET, "wb") as local_file:
                shutil.copyfileobj(web_file, local_file)

    # Count the lines, minus 1 for the header row.
    rows = sum(1 for _ in TARGET.open()) - 1
    print(f"Done. {rows} rows saved to {TARGET}")


if __name__ == "__main__":
    main()
