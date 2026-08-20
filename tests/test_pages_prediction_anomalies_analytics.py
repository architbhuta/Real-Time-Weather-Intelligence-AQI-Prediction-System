from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves a relative script_path against the directory of
# the *calling test file* (tests/), not against pytest's cwd/rootdir. Repo-root
# scripts therefore need absolute paths here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_prediction_page_runs_without_exception():
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "3_Prediction.py"))
    at.run(timeout=30)

    assert not at.exception
    assert any("AQI Prediction" in str(el.value) for el in at.title)


def test_anomalies_page_runs_without_exception():
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "4_Anomalies.py"))
    at.run(timeout=30)

    assert not at.exception
    assert any("Anomaly Detection" in str(el.value) for el in at.title)


def test_historical_analytics_page_runs_without_exception():
    at = AppTest.from_file(str(PROJECT_ROOT / "pages" / "5_Historical_Analytics.py"))
    at.run(timeout=30)

    assert not at.exception
    assert any("Historical Analytics" in str(el.value) for el in at.title)
