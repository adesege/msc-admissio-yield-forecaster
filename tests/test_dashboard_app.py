import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/overview.py", "views/explain.py", "views/what_if.py", "views/fairness.py", "views/about.py"]


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
