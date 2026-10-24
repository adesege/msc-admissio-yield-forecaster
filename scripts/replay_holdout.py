import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

HOLDOUT = Path(__file__).resolve().parents[1] / "dashboard" / "data" / "holdout.parquet"
FIELDS = ["applicants", "offers", "last_cycle_applicants", "last_cycle_offers", "last_cycle_yield", "test_policy",
          "last_cycle_test_policy", "control", "size_band", "locale", "region", "carnegie_class", "hbcu"]


def request_body(row: pd.Series) -> dict:
    body = {"institution_id": row["unitid"]}
    for field in FIELDS:
        value = row[field]
        body[field] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
    return body


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(f"{url}/predict", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send the 2024 hold-out cycle through the API as live traffic.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    holdout = pd.read_parquet(HOLDOUT)
    if args.limit:
        holdout = holdout.head(args.limit)
    alerts = 0
    for _, row in holdout.iterrows():
        alerts += post(args.url, request_body(row))["yield_drop_alert"]
    print(f"Sent {len(holdout):,} forecasts; {alerts:,} raised a yield-drop alert.")


if __name__ == "__main__":
    main()
