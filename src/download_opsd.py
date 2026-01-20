from pathlib import Path
import urllib.request

from config import DATA_DIR, OPSD_URL, OPSD_PATH

def download_if_missing(url, dest):
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Skipping download if file already exists
    if dest.exists():
        print(f"File already exists: {dest}")
        return

    print(f"Downloading OPSD data to {dest} ..")
    urllib.request.urlretrieve(url, dest)
    print("Download complete!")

if __name__ == "__main__":
    download_if_missing(OPSD_URL, OPSD_PATH)