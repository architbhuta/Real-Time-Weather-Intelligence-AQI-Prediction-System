"""IQR-based detection of statistically unusual readings.

A value is flagged when it falls outside the Tukey fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
of that metric's full recorded history for the location.

Everything this module reports is *statistical*: a flagged reading is unusual
compared with what this location has recorded before, and nothing more. It is
explicitly **not** a health, safety or danger assessment — an ordinary-looking day
in a heavily polluted city can be harmful while raising no flag here, and a clean
day after a run of dirty ones can be flagged while being perfectly safe to breathe.
Never present these results as a hazard warning.
"""

import pandas as pd

METRICS = ["pm25", "pm10", "aqi", "temperature", "humidity"]


def compute_iqr_bounds(values: pd.Series) -> tuple[float, float, float]:
    """Return the (lower_bound, upper_bound, iqr) Tukey fence for a series of values."""
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper, iqr


def classify_severity(distance_ratio: float) -> str:
    """Grade how far outside the IQR fence a value sits: "Low", "Medium" or "High".

    ``distance_ratio`` is the distance from the nearest fence, measured in IQRs, so
    the grades are purely statistical: "Low" is under 1 IQR beyond the fence, "Medium"
    under 3, "High" beyond that.

    This severity carries **no health or danger meaning whatsoever**. "High" means
    "far from this location's usual spread", not "hazardous"; "Low" does not mean
    "safe". Air that is dangerous every single day is, statistically, unremarkable.
    """
    if distance_ratio < 1.0:
        return "Low"
    if distance_ratio < 3.0:
        return "Medium"
    return "High"


def detect_anomalies_for_metric(df: pd.DataFrame, metric: str, min_history: int = 30) -> list[dict]:
    """Flag readings of one metric that fall outside its historical IQR fence.

    Returns one record per flagged reading. Note that two of the numeric fields are
    on entirely different scales and are not comparable with each other:

    - ``observed_value`` and ``expected_value`` are in the metric's own units;
      ``expected_value`` is the historical *median*.
    - ``anomaly_score`` is a *distance beyond the fence, measured in IQRs* — not in
      the metric's units, and not a distance from the median.

    So ``observed=213.7, expected=75.4, score=0.02`` is consistent, not a bug: the
    reading is ~3x the median, but the IQR is wide, which puts the fence itself far
    out, and the reading only just clears it.

    ``min_history`` is the smallest number of non-null readings worth computing a
    distribution from; below it nothing is flagged.

    Expects ``df`` to have a unique index (as ``load_weather_and_air_quality``
    guarantees via ``reset_index(drop=True)``) so that each positional lookup of
    ``timestamp`` yields a scalar rather than a Series.
    """
    values = df[metric].dropna()
    if len(values) < min_history:
        return []

    lower, upper, iqr = compute_iqr_bounds(values)
    if iqr == 0:
        return []

    expected = float(values.median())
    anomalies = []
    for idx, value in values.items():
        if value < lower:
            distance_ratio = (lower - value) / iqr
        elif value > upper:
            distance_ratio = (value - upper) / iqr
        else:
            continue

        anomalies.append({
            "timestamp": df["timestamp"].loc[idx],
            "metric": metric,
            "observed_value": round(float(value), 2),
            "expected_value": round(expected, 2),
            "anomaly_score": round(float(distance_ratio), 2),
            "severity": classify_severity(distance_ratio),
        })
    return anomalies


def detect_anomalies_for_location(
    df: pd.DataFrame, location: str, min_history: int = 30
) -> list[dict]:
    """Run :func:`detect_anomalies_for_metric` over every metric present in ``df``.

    Returns records ready for ``data.database.insert_anomalies``. ``min_history`` is
    passed through to each metric's detection unchanged.
    """
    all_anomalies = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        for record in detect_anomalies_for_metric(df, metric, min_history=min_history):
            record["location"] = location
            all_anomalies.append(record)
    return all_anomalies
