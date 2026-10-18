import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import default_paths

RAW_DIR = default_paths().raw_dir

ADMISSIONS_FILES = {
    2022: "adm2022_rv.csv",
    2023: "adm2023_rv.csv",
    2024: "adm2024.csv",
}
DIRECTORY_FILE = "hd2023.csv"

FUNNEL_COLUMNS = ["APPLCN", "ADMSSN", "ENRLT"]
GENDER_SPLIT_COLUMNS = {
    "APPLCN": ["APPLCNM", "APPLCNW", "APPLCNAN", "APPLCNUN"],
    "ADMSSN": ["ADMSSNM", "ADMSSNW", "ADMSSNAN", "ADMSSNUN"],
    "ENRLT": ["ENRLM", "ENRLW", "ENRLAN", "ENRLUN"],
}
ADMISSIONS_COLUMNS = ["UNITID", *FUNNEL_COLUMNS, "ADMCON7"]
DIRECTORY_COLUMNS = [
    "UNITID", "INSTNM", "STABBR", "OBEREG", "SECTOR", "CONTROL",
    "HBCU", "TRIBAL", "LOCALE", "INSTSIZE", "C21BASIC",
]


@dataclass(frozen=True)
class LandedFile:
    name: str
    path: Path
    sha256: str
    size_bytes: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def land_raw_files(raw_dir: Path = RAW_DIR) -> list[LandedFile]:
    expected = [*ADMISSIONS_FILES.values(), DIRECTORY_FILE]
    missing = [name for name in expected if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Raw IPEDS files missing from {raw_dir}: {', '.join(missing)}")
    return [
        LandedFile(name, raw_dir / name, file_sha256(raw_dir / name), (raw_dir / name).stat().st_size)
        for name in expected
    ]


def read_ipeds_csv(path: Path) -> pd.DataFrame:
    # NCES files mix UTF-8 with a byte-order mark and Latin-1, and some headers carry trailing spaces.
    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def load_admissions(raw_dir: Path = RAW_DIR, years=None, include_gender_splits: bool = False) -> pd.DataFrame:
    columns = list(ADMISSIONS_COLUMNS)
    if include_gender_splits:
        columns += [col for parts in GENDER_SPLIT_COLUMNS.values() for col in parts]
    numeric_columns = [col for col in columns if col != "UNITID"]
    frames = []
    for year in years or ADMISSIONS_FILES:
        df = read_ipeds_csv(raw_dir / ADMISSIONS_FILES[year])[columns].copy()
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        df["YEAR"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_directory(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return read_ipeds_csv(raw_dir / DIRECTORY_FILE)[DIRECTORY_COLUMNS].copy()
