import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/overview.py", "views/explain.py", "views/what_if.py", "views/planner.py", "views/fairness.py",
         "views/about.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    app = AppTest.from_file("../dashboard/app.py", default_timeout=120)
    app.run()
    app.switch_page(page).run()
    assert not app.exception
    assert app.title[0].value


def test_what_if_reports_change_against_reported_offers():
    app = AppTest.from_file("../dashboard/app.py", default_timeout=120)
    app.run()
    app.switch_page("views/what_if.py").run()
    offers = app.slider[1]
    offers.set_value(offers.value // 2).run()
    assert not app.exception
    assert app.metric[1].delta.startswith("-")


def _page(page: str) -> AppTest:
    app = AppTest.from_file("../dashboard/app.py", default_timeout=180)
    app.run()
    return app.switch_page(page).run()


def test_explain_runs_lime_and_counterfactuals():
    app = _page("views/explain.py")
    app.button[0].click().run()
    assert not app.exception
    assert any("local model explains" in caption.value for caption in app.caption)
    app.button[1].click().run()
    assert not app.exception


def test_planner_keeps_upper_bound_within_capacity():
    app = _page("views/planner.py")
    assert not app.exception
    suggested = int(app.metric[0].value.replace(",", ""))
    upper = int(app.metric[2].value.split(" to ")[1].replace(",", ""))
    assert suggested > 0
    assert upper <= app.number_input[0].value


def test_planner_warns_when_capacity_is_too_small():
    app = _page("views/planner.py")
    app.number_input[0].set_value(1).run()
    assert not app.exception
    assert app.error
