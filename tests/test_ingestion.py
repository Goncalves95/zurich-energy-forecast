"""Tests for ingestion layer."""
from unittest.mock import MagicMock, patch

import pandas as pd

CKAN_PACKAGE_RESPONSE = {
    "result": {
        "resources": [
            {"format": "CSV", "url": "https://example.org/ewz.csv"},
        ]
    }
}

SAMPLE_CSV = (
    "Timestamp,Value_NE5,Value_NE7\n"
    "2015-01-01T00:00:00Z,100.0,200.0\n"
    "2015-01-01T00:15:00Z,,150.0\n"  # missing NE5, still valid (NE7 present)
    "not-a-date,50.0,60.0\n"  # malformed timestamp, should be skipped
    "2015-01-01T00:30:00Z,,\n"  # both values missing, should be skipped
)


def _mock_requests_get(package_response, csv_text):
    """Return a side_effect for requests.get: first call hits CKAN, second downloads the CSV."""
    package_resp = MagicMock()
    package_resp.json.return_value = package_response
    package_resp.raise_for_status.return_value = None

    csv_resp = MagicMock()
    csv_resp.text = csv_text
    csv_resp.raise_for_status.return_value = None

    return [package_resp, csv_resp]


def _recent_local_timestamp(hours_ago: int = 1) -> str:
    """A timestamp string in MeteoSwiss's YYYYMMDDHHMM local-time format, recent
    enough to pass the ingestion lookback cutoff."""
    ts = pd.Timestamp.now(tz="Europe/Zurich") - pd.Timedelta(hours=hours_ago)
    return ts.strftime("%Y%m%d%H%M")


def _weather_csv() -> str:
    recent = _recent_local_timestamp(hours_ago=1)
    return (
        "Station/Location;Date;tre200s0;rre150z0;sre000z0;gre000z0;ure200s0\n"
        f"SMA;{recent};5.20;0.00;0.00;120.00;80.50\n"  # SMA, fully populated
        f"KLO;{recent};3.10;0.00;0.00;100.00;75.00\n"  # different station, filtered out
        f"SMA;notadate;9.99;0.00;0.00;999.00;99.00\n"  # malformed timestamp, skipped
        f"SMA;{recent};-;0.00;0.00;-;-\n"  # SMA, missing sensor values but valid timestamp
    )


@patch("ingestion.fetch_energy.execute_values")
@patch("ingestion.fetch_energy.get_connection")
@patch("ingestion.fetch_energy.requests.get")
def test_fetch_energy_inserts_parsed_rows(mock_get, mock_get_connection, mock_execute_values):
    from ingestion.fetch_energy import fetch_energy

    mock_get.side_effect = _mock_requests_get(CKAN_PACKAGE_RESPONSE, SAMPLE_CSV)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"max_ts": None}
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value = mock_conn

    # Simulate every row being newly inserted (no conflicts).
    mock_execute_values.side_effect = lambda cur, sql, rows, **kw: [{"id": i} for i in range(len(rows))]

    fetch_energy(full_load=True)

    mock_execute_values.assert_called_once()
    _, sql, rows = mock_execute_values.call_args.args
    kwargs = mock_execute_values.call_args.kwargs
    assert "INSERT INTO raw_energy" in sql
    assert "ON CONFLICT (timestamp, source) DO NOTHING" in sql
    assert kwargs["page_size"] == 1000
    assert kwargs["fetch"] is True

    # 2 valid rows (malformed timestamp and all-missing-values rows are skipped)
    assert len(rows) == 2
    inserted_kwh = {row[1] for row in rows}
    assert inserted_kwh == {300.0, 150.0}
    mock_conn.commit.assert_called()
    mock_conn.close.assert_called_once()


@patch("ingestion.fetch_energy.execute_values")
@patch("ingestion.fetch_energy.get_connection")
@patch("ingestion.fetch_energy.requests.get")
def test_fetch_energy_skips_duplicates(mock_get, mock_get_connection, mock_execute_values):
    from ingestion.fetch_energy import fetch_energy

    mock_get.side_effect = _mock_requests_get(CKAN_PACKAGE_RESPONSE, SAMPLE_CSV)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"max_ts": None}
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value = mock_conn

    # Simulate every row already existing (ON CONFLICT DO NOTHING returns nothing).
    mock_execute_values.return_value = []

    fetch_energy(full_load=True)

    mock_execute_values.assert_called_once()
    _, sql, rows = mock_execute_values.call_args.args
    assert len(rows) == 2
    mock_conn.commit.assert_called()


@patch("ingestion.fetch_energy.get_connection")
@patch("ingestion.fetch_energy.requests.get")
def test_fetch_energy_network_error_propagates(mock_get, mock_get_connection):
    import requests

    from ingestion.fetch_energy import fetch_energy

    mock_get.side_effect = requests.exceptions.ConnectionError("boom")

    try:
        fetch_energy()
        assert False, "expected ConnectionError to propagate"
    except requests.exceptions.ConnectionError:
        pass

    mock_get_connection.assert_not_called()


@patch("ingestion.fetch_weather.execute_values")
@patch("ingestion.fetch_weather.get_connection")
@patch("ingestion.fetch_weather.requests.get")
def test_fetch_weather_filters_station_and_inserts(mock_get, mock_get_connection, mock_execute_values):
    from ingestion.fetch_weather import fetch_weather

    mock_response = MagicMock()
    mock_response.text = _weather_csv()
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn

    # Simulate every row being newly inserted (no conflicts).
    mock_execute_values.side_effect = lambda cur, sql, rows, **kw: [{"id": i} for i in range(len(rows))]

    fetch_weather(full_load=True)

    mock_get.assert_called_once()
    mock_execute_values.assert_called_once()
    _, sql, rows = mock_execute_values.call_args.args
    kwargs = mock_execute_values.call_args.kwargs
    assert "INSERT INTO raw_weather" in sql
    assert "ON CONFLICT (timestamp, source) DO NOTHING" in sql
    assert kwargs["page_size"] == 1000
    assert kwargs["fetch"] is True

    # Only the 2 SMA rows with a parseable timestamp remain: KLO is filtered out
    # by station, and the "notadate" row is dropped as malformed.
    assert len(rows) == 2
    temperatures = {row[1] for row in rows}
    assert temperatures == {5.20, None}
    mock_conn.commit.assert_called()
    mock_conn.close.assert_called_once()


@patch("ingestion.fetch_weather.execute_values")
@patch("ingestion.fetch_weather.get_connection")
@patch("ingestion.fetch_weather.requests.get")
def test_fetch_weather_skips_duplicates(mock_get, mock_get_connection, mock_execute_values):
    from ingestion.fetch_weather import fetch_weather

    mock_response = MagicMock()
    mock_response.text = _weather_csv()
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn

    # Simulate every row already existing (ON CONFLICT DO NOTHING returns nothing).
    mock_execute_values.return_value = []

    fetch_weather(full_load=True)

    mock_execute_values.assert_called_once()
    _, sql, rows = mock_execute_values.call_args.args
    assert len(rows) == 2
    mock_conn.commit.assert_called()


@patch("ingestion.fetch_weather.get_connection")
@patch("ingestion.fetch_weather.requests.get")
def test_fetch_weather_network_error_propagates(mock_get, mock_get_connection):
    import requests

    from ingestion.fetch_weather import fetch_weather

    mock_get.side_effect = requests.exceptions.ConnectionError("boom")

    try:
        fetch_weather()
        assert False, "expected ConnectionError to propagate"
    except requests.exceptions.ConnectionError:
        pass

    mock_get_connection.assert_not_called()
