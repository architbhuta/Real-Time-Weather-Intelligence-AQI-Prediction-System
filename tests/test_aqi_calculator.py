# tests/test_aqi_calculator.py
from data.aqi_calculator import aqi_category, calculate_cpcb_aqi


def test_calculate_cpcb_aqi_pm25_only():
    # PM2.5 = 15 falls in the 0-30 -> 0-50 bracket: Ip = (50/30)*15 = 25
    aqi, dominant = calculate_cpcb_aqi(pm25=15, pm10=None, co=None, no2=None, so2=None, o3=None)
    assert aqi == 25
    assert dominant == "pm25"


def test_calculate_cpcb_aqi_takes_worst_pollutant():
    # PM2.5=15 (-> 25), PM10=300 (301-350 -> 201-300 bracket, high sub-index)
    aqi, dominant = calculate_cpcb_aqi(pm25=15, pm10=300, co=None, no2=None, so2=None, o3=None)
    assert dominant == "pm10"
    assert aqi > 25


def test_calculate_cpcb_aqi_returns_none_when_no_pollutants():
    aqi, dominant = calculate_cpcb_aqi(pm25=None, pm10=None, co=None, no2=None, so2=None, o3=None)
    assert aqi is None
    assert dominant is None


def test_aqi_category_boundaries():
    assert aqi_category(30) == "Good"
    assert aqi_category(75) == "Satisfactory"
    assert aqi_category(150) == "Moderate"
    assert aqi_category(250) == "Poor"
    assert aqi_category(350) == "Very Poor"
    assert aqi_category(450) == "Severe"
