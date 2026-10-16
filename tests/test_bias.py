import pandas as pd

from src.fairness.representation import flagged_groups, representation_report, representation_table


def test_group_losing_share_is_flagged():
    panel = pd.DataFrame({"CONTROL_GROUP": ["Public"] * 100 + ["Private for-profit"] * 100})
    eligible = pd.Series([True] * 100 + [True] * 50 + [False] * 50)
    table = representation_table(panel, eligible, "CONTROL_GROUP").set_index("group")
    assert table.loc["Private for-profit", "status"] == "under-represented"
    assert table.loc["Public", "representation_ratio"] > 1


def test_groups_below_minimum_size_are_unassessed():
    panel = pd.DataFrame({"TRIBAL_GROUP": ["Tribal college"] * 5 + ["Not tribal college"] * 95})
    table = representation_table(panel, pd.Series([True] * 100), "TRIBAL_GROUP").set_index("group")
    assert table.loc["Tribal college", "status"].startswith("unassessed")


def test_real_panel_flags_for_profit_attrition(yield_panel):
    report = representation_report(yield_panel, yield_panel["FULL_YIELD_HISTORY"])
    assert "Private for-profit" in flagged_groups(report)
