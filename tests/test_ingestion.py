"""Tests for ingestion layer."""


def test_fetch_energy_returns_none():
    from ingestion.fetch_energy import fetch_energy
    assert fetch_energy() is None


def test_fetch_weather_returns_none():
    from ingestion.fetch_weather import fetch_weather
    assert fetch_weather() is None
