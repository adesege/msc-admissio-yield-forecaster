from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "ipeds"

ADMISSIONS_FILES = {
    2022: "adm2022_rv.csv",
    2023: "adm2023_rv.csv",
    2024: "adm2024.csv",
}
DIRECTORY_FILE = "hd2023.csv"

FUNNEL_COLUMNS = ["APPLCN", "ADMSSN", "ENRLT"]
ADMISSIONS_COLUMNS = ["UNITID", *FUNNEL_COLUMNS, "ADMCON7"]
DIRECTORY_COLUMNS = [
    "UNITID", "INSTNM", "STABBR", "OBEREG", "SECTOR", "CONTROL",
    "HBCU", "TRIBAL", "LOCALE", "INSTSIZE", "C21BASIC",
]


def read_ipeds_csv(path: Path) -> pd.DataFrame:
    # NCES files mix UTF-8 with a byte-order mark and Latin-1, and some headers carry trailing spaces.
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def load_admissions(raw_dir: Path = RAW_DIR, years=None) -> pd.DataFrame:
    frames = []
    for year in years or ADMISSIONS_FILES:
        df = read_ipeds_csv(raw_dir / ADMISSIONS_FILES[year])[ADMISSIONS_COLUMNS].copy()
        for col in [*FUNNEL_COLUMNS, "ADMCON7"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df["YEAR"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_directory(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return read_ipeds_csv(raw_dir / DIRECTORY_FILE)[DIRECTORY_COLUMNS].copy()
