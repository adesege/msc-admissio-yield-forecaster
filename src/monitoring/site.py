import argparse
import html
import json
import shutil
from pathlib import Path

from src.config import default_paths
from src.models.features import FEATURE_LABELS

REPOSITORY = "https://github.com/adesege/msc-admissio-yield-forecaster"
DASHBOARD = "https://msc-admissio-yield-forecaster.streamlit.app/"
SIGNAL_LABELS = {
    "feature_psi": "Input drift (PSI)",
    "yield_mae": "Yield error (MAE, points)",
    "error_ratio": "Median error ratio vs other institutions",
    "error_ratio_change": "Change in error ratio since last cycle",
    "interval_coverage": "80% interval coverage",
    "alert_false_positive_multiple": "Yield-drop false alarms vs overall rate",
    "missing_input_share": "Rows with a missing input",
    "representation_ratio": "Representation in the modelling set",
    "capacity_overrun_share": "Offer planner: intake above capacity",
}
PERCENT_SIGNALS = {"interval_coverage", "missing_input_share", "capacity_overrun_share"}

STYLE = """
:root{--bg:#fbfaf7;--card:#fff;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--green:#1baf7a;--amber:#b88400;--red:#d03b3b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#f2f1ec;--muted:#b4b2aa;--line:#34332f}
img{background:#fff;border-radius:6px}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
p{margin:6px 0 10px}.muted{color:var(--muted)}a{color:inherit}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:22px}.tile span{color:var(--muted);font-size:13px}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:600;color:#fff}
.status-green{color:var(--green)}.status-amber{color:var(--amber)}.status-red{color:var(--red)}
.green{background:var(--green)}.amber{background:var(--amber)}.red{background:var(--red)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.scroll{overflow-x:auto}figure{margin:14px 0}img{max-width:100%;height:auto}figcaption{color:var(--muted);font-size:13px}
details{margin:12px 0}summary{cursor:pointer;color:var(--muted)}
.banner{border-left:4px solid var(--red);background:var(--card);padding:10px 14px;border-radius:6px;margin:14px 0}
"""


def _value(signal: str, value) -> str:
    if value is None:
        return ""
    return f"{100 * value:.1f}%" if signal in PERCENT_SIGNALS else f"{value:.3g}" if abs(value) < 1 else f"{value:.2f}"


def _group(check: dict) -> str:
    group = check["group"] or ""
    return FEATURE_LABELS.get(group, group)


def check_rows(checks: list[dict]) -> str:
    rows = []
    for check in sorted(checks, key=lambda c: (c["signal"], _group(c))):
        rows.append(
            f"<tr><td><span class='pill {check['status']}'>{check['status']}</span></td>"
            f"<td>{html.escape(SIGNAL_LABELS.get(check['signal'], check['signal']))}</td>"
            f"<td>{html.escape(_group(check))}</td><td>{_value(check['signal'], check['value'])}</td>"
            f"<td class='muted'>{html.escape(check['rule'])}</td></tr>"
        )
    return "".join(rows)


def render(summary: dict) -> str:
    decision = summary["decision"]
    model = summary["performance"]["model"]
    baseline = summary["performance"]["baseline"]
    planner = next(r for r in summary["offer_planner"]["by_group"] if r["group"] == "All institutions")
    red = len(decision["red_signals"])
    actions = []
    if decision["panel_review_required"]:
        actions.append("Fairness or alert signals are red, so an incident is open for the Ethics Review Panel.")
    if decision["retrain_required"]:
        actions.append("Drift or error is red, so the retraining workflow has produced a challenger.")
    banner = (f"<div class='banner'><b>{red} red signal{'s' if red != 1 else ''}.</b> "
              f"{' '.join(actions)}</div>") if red else ""
    release = "approved for release" if summary["approved_for_release"] else "candidate, not approved for release"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admissio Model Monitoring</title><style>{STYLE}</style></head><body><main>
<h1>Admissio Yield Forecaster monitoring</h1>
<p class="muted">Model {summary['model_version']} ({release}). Reference cycle {summary['reference_cycle']}, current cycle
{summary['current_cycle']}, {summary['institutions']['current']:,} institutions. Generated {summary['generated_at']}.</p>
{banner}
<div class="tiles">
<div class="tile"><span>Overall status</span><b class="status-{decision['overall_status']}">{decision['overall_status'].capitalize()}</b></div>
<div class="tile"><span>Yield MAE (baseline {baseline['yield_mae_points']:.2f})</span><b>{model['yield_mae_points']:.2f} pts</b></div>
<div class="tile"><span>Largest input drift (PSI)</span><b>{max(summary['psi'].values()):.3f}</b></div>
<div class="tile"><span>Alert disparate impact</span><b>{summary['alerts']['summary']['disparate_impact_ratio']:.2f}</b></div>
<div class="tile"><span>Planner overrun (naive {planner['naive_overrun_share']:.0%})</span><b>{planner['planner_overrun_share']:.1%}</b></div>
</div>
<h2>Signals needing attention</h2>
<p class="muted">Thresholds come from the model card. Red opens an incident; drift or error reds also start retraining.</p>
<div class="scroll"><table><thead><tr><th>Status</th><th>Signal</th><th>Group or input</th><th>Value</th><th>Rule</th></tr></thead>
<tbody>{check_rows([c for c in summary['checks'] if c['status'] != 'green'])}</tbody></table></div>
<details><summary>All {len(summary['checks'])} checks</summary><div class="scroll"><table><thead><tr><th>Status</th><th>Signal</th>
<th>Group or input</th><th>Value</th><th>Rule</th></tr></thead><tbody>{check_rows(summary['checks'])}</tbody></table></div></details>
<h2>Fairness drift</h2>
<figure><img src="figures/error_ratios.png" alt="Median error ratio by group, previous and current cycle">
<figcaption>Bias drift: each group's median forecast error relative to all other institutions, cycle on cycle.</figcaption></figure>
<figure><img src="figures/coverage_alerts.png" alt="Interval coverage and false-alarm multiples by group">
<figcaption>Left: share of actual outcomes inside the 80% range (shaded band 75% to 85%). Right: false-alarm rate of the
yield-drop alert relative to the overall rate.</figcaption></figure>
<h2>Data drift</h2>
<figure><img src="figures/psi.png" alt="Population stability index by input">
<figcaption>Population stability index for every model input.</figcaption></figure>
<p>Evidently reports: <a href="evidently/data_drift.html">data drift</a> ({len(summary['evidently']['data_drift']['drifted_columns'])}
of {len(summary['psi'])} inputs flagged) and <a href="evidently/regression.html">regression performance</a>.</p>
<h2>Offer planner outcome</h2>
<figure><img src="figures/planner.png" alt="Share of institutions over capacity by group">
<figcaption>Offers sized so the upper 80% bound on intake meets capacity, replayed against the 2024 outcome. Median fill
was {planner['planner_median_fill']:.0%} of capacity against {planner['naive_median_fill']:.0%} for the naive plan.</figcaption></figure>
<p class="muted">Stakeholder dashboard: <a href="{DASHBOARD}">{DASHBOARD}</a>. Source and thresholds:
<a href="{REPOSITORY}/blob/main/docs/model_card.md">model card</a>,
<a href="{REPOSITORY}/blob/main/docs/monitoring_plan.md">monitoring plan</a>.</p>
</main></body></html>
"""


def build(report_dir: Path, site_dir: Path) -> Path:
    summary = json.loads((report_dir / "summary.json").read_text())
    if site_dir.exists():
        shutil.rmtree(site_dir)
    shutil.copytree(report_dir / "figures", site_dir / "figures")
    shutil.copytree(report_dir / "evidently", site_dir / "evidently")
    shutil.copy2(report_dir / "summary.json", site_dir / "summary.json")
    (site_dir / "index.html").write_text(render(summary))
    return site_dir / "index.html"


def incident_body(summary: dict) -> str:
    lines = [
        f"Monitoring run {summary['generated_at']} on model {summary['model_version']} "
        f"(cycle {summary['current_cycle']}) found red signals.",
        "",
        "| Signal | Group or input | Value | Rule |",
        "|---|---|---|---|",
    ]
    for check in summary["decision"]["red_signals"]:
        lines.append(f"| {SIGNAL_LABELS.get(check['signal'], check['signal'])} | {_group(check)} | "
                     f"{_value(check['signal'], check['value'])} | {check['rule']} |")
    lines += [
        "",
        "Follow `docs/incident_response.md`: triage within two working days, decide whether partner-facing outputs "
        "are paused, and record the Ethics Review Panel decision on this issue before closing it.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the monitoring site or an incident issue body.")
    parser.add_argument("--report-dir", type=Path, default=default_paths().reports_dir / "monitoring")
    parser.add_argument("--site-dir", type=Path, default=default_paths().root / "site")
    parser.add_argument("--incident-body", type=Path, help="Write a GitHub issue body for the red signals here.")
    arguments = parser.parse_args()
    if arguments.incident_body:
        arguments.incident_body.write_text(incident_body(json.loads((arguments.report_dir / "summary.json").read_text())))
    else:
        print(build(arguments.report_dir, arguments.site_dir))
