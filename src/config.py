from pathlib import Path

DATA_DIR = Path("data")
# Add the URL here
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
# The corresponding file name
OPSD_FILENAME = "time_series_60min_singleindex.csv"
# Finally, the corresponding path
OPSD_PATH = DATA_DIR / OPSD_FILENAME