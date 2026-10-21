import argparse
import json

import httpx

from dashboard.core import EXAMPLE_LABELS, PLAIN_LABELS, LABEL_TO_FEATURE, cycle_from_row, load_dashboard_data


def explain(label: str, request: dict, response: dict) -> str:
    interval = response["yield_interval_80"]
    enrolled = response["expected_enrolled_interval_80"]
    drivers = "; ".join(
        f"{PLAIN_LABELS[LABEL_TO_FEATURE[d['feature']]]} {d['contribution_points']:+.1f} points"
        for d in response["top_drivers"]
    )
    return "\n".join(
        [
            f"{label} ({response['fairness_group']})",
            f"  Forecast yield {response['forecast_yield']:.1%}, likely range {interval['lower']:.0%} to "
            f"{interval['upper']:.0%}",
            f"  With {request['offers']:,} offers: about {response['expected_enrolled']:,} new students "
            f"({enrolled['lower']:,.0f} to {enrolled['upper']:,.0f})",
            f"  Yield-drop alert: {'raised' if response['yield_drop_alert'] else 'not raised'}",
            f"  Largest drivers: {drivers}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send the three sample institutions to the forecast API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--raw", action="store_true", help="Print the JSON request and response for each call.")
    args = parser.parse_args()

    data = load_dashboard_data()
    with httpx.Client(base_url=args.url, timeout=30) as client:
        health = client.get("/health").raise_for_status().json()
        print(f"API {health['status']}: model {health['model_version']}, "
              f"approved for release: {health['approved_for_release']}\n")
        for key, unitid in data.summary["examples"].items():
            request = cycle_from_row(data.institution(unitid)).model_dump()
            response = client.post("/predict", json=request).raise_for_status().json()
            if args.raw:
                print(json.dumps({"request": request, "response": response}, indent=2))
            print(explain(EXAMPLE_LABELS[key], request, response), end="\n\n")
        print(response["notice"])


if __name__ == "__main__":
    main()
