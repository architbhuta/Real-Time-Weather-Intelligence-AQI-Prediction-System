"""CPCB National Air Quality Index calculation.

Breakpoints are from CPCB's National AQI (2014) sub-index tables, in
µg/m3 for PM2.5, PM10, NO2, SO2, O3 (24-hr avg, O3 8-hr avg) and mg/m3
for CO (8-hr avg). Overall AQI is the maximum of the available
sub-indices, per CPCB methodology, and is only reported when CPCB's
minimum-data rule is met: at least three pollutants, including PM2.5 or
PM10. Open-Meteo returns CO in µg/m3, so it is converted to mg/m3
before lookup.

Each tuple is (concentration_low, concentration_high, index_low, index_high).
"""

PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (30, 60, 50, 100),
    (60, 90, 100, 200),
    (90, 120, 200, 300),
    (120, 250, 300, 400),
    (250, 380, 400, 500),
]
PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 250, 100, 200),
    (250, 350, 200, 300),
    (350, 430, 300, 400),
    (430, 510, 400, 500),
]
NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (40, 80, 50, 100),
    (80, 180, 100, 200),
    (180, 280, 200, 300),
    (280, 400, 300, 400),
    (400, 800, 400, 500),
]
SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (40, 80, 50, 100),
    (80, 380, 100, 200),
    (380, 800, 200, 300),
    (800, 1600, 300, 400),
    (1600, 2100, 400, 500),
]
O3_BREAKPOINTS = [
    (0, 50, 0, 50),
    (50, 100, 50, 100),
    (100, 168, 100, 200),
    (168, 208, 200, 300),
    (208, 748, 300, 400),
    (748, 1000, 400, 500),
]
CO_BREAKPOINTS_MG = [
    (0, 1.0, 0, 50),
    (1.0, 2.0, 50, 100),
    (2.0, 10.0, 100, 200),
    (10.0, 17.0, 200, 300),
    (17.0, 34.0, 300, 400),
    (34.0, 50.0, 400, 500),
]

# CPCB validity threshold: a National AQI needs at least three pollutants,
# and at least one of them must be PM2.5 or PM10.
MINIMUM_POLLUTANTS = 3
PARTICULATE_POLLUTANTS = frozenset({"pm25", "pm10"})

CATEGORY_THRESHOLDS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
]


def _sub_index(concentration: float, breakpoints: list[tuple[float, float, float, float]]) -> float:
    if concentration <= 0:
        concentration = 0
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    # above the top bracket: clamp to the max index
    return breakpoints[-1][3]


def calculate_cpcb_aqi(
    pm25: float | None,
    pm10: float | None,
    co: float | None,
    no2: float | None,
    so2: float | None,
    o3: float | None,
) -> tuple[int | None, str | None]:
    sub_indices: dict[str, float] = {}
    if pm25 is not None:
        sub_indices["pm25"] = _sub_index(pm25, PM25_BREAKPOINTS)
    if pm10 is not None:
        sub_indices["pm10"] = _sub_index(pm10, PM10_BREAKPOINTS)
    if no2 is not None:
        sub_indices["no2"] = _sub_index(no2, NO2_BREAKPOINTS)
    if so2 is not None:
        sub_indices["so2"] = _sub_index(so2, SO2_BREAKPOINTS)
    if o3 is not None:
        sub_indices["o3"] = _sub_index(o3, O3_BREAKPOINTS)
    if co is not None:
        sub_indices["co"] = _sub_index(co / 1000.0, CO_BREAKPOINTS_MG)

    # CPCB requires at least three pollutants, one of which must be PM2.5 or
    # PM10, before a National AQI is considered valid. A degraded API response
    # below that bar yields no AQI rather than a falsely confident one.
    if len(sub_indices) < MINIMUM_POLLUTANTS or not (PARTICULATE_POLLUTANTS & sub_indices.keys()):
        return None, None

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    return round(sub_indices[dominant_pollutant]), dominant_pollutant


def aqi_category(aqi: int) -> str:
    for threshold, category in CATEGORY_THRESHOLDS:
        if aqi <= threshold:
            return category
    return "Severe"
