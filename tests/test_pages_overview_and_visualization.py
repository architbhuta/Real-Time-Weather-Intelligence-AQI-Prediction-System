from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves a relative script_path against the directory of
# the *calling test file* (tests/), not against pytest's cwd/rootdir. Repo-root
# scripts therefore need absolute paths here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_app_entry_runs_without_exception():
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    at.run(timeout=30)

    assert not at.exception
    assert any("AirSense" in str(el.value) for el in at.title)


def test_live_overview_page_runs_without_exception():
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "1_Live_Overview.py"))
    at.run(timeout=30)

    assert not at.exception
    assert any("Live Overview" in str(el.value) for el in at.title)


def test_visualization_page_runs_without_exception():
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "2_Visualization.py"))
    at.run(timeout=30)

    assert not at.exception
    assert any("Live Visualization" in str(el.value) for el in at.title)
