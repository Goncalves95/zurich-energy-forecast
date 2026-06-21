"""Tests for pipeline transformations."""
from unittest.mock import MagicMock, patch

import pandas as pd


def _mock_conn_with_rows(*fetchall_results):
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = list(fetchall_results)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn


# ---------------------------------------------------------------------------
# bronze_to_silver
# ---------------------------------------------------------------------------

def test_clean_energy_dedupes_outliers_and_fills_short_gaps():
    from pipeline.bronze_to_silver import clean_energy

    ts = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": list(ts[:6]) + [ts[1]],  # duplicate of ts[1]
        "kwh": [100, 200, -5, 999_999, 150, 160, 999],
    })

    out = clean_energy(df)

    # ts[1] duplicate resolves to the later value (999); ts[2]/ts[3] outliers
    # (<=0 and >180000) become a 2h gap that gets forward-filled from 999.
    assert list(out["kwh"]) == [100.0, 999.0, 999.0, 999.0, 150.0, 160.0]
    assert list(out["timestamp"]) == list(ts[:6])


def test_clean_energy_drops_gaps_longer_than_limit():
    from pipeline.bronze_to_silver import clean_energy

    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    keep = [0, 1, 6, 7, 8, 9]  # hours 2..5 missing -> 4h gap, limit is 3h
    df = pd.DataFrame({"timestamp": [ts[i] for i in keep], "kwh": [100, 110, 160, 170, 180, 190]})

    out = clean_energy(df)

    # hour 5 is the 4th consecutive missing hour, beyond the fill limit, so it's dropped.
    kept_timestamps = set(out["timestamp"])
    assert ts[5] not in kept_timestamps
    assert {ts[2], ts[3], ts[4]} <= kept_timestamps
    assert len(out) == 9


def test_clean_weather_clamps_ranges_and_drops_all_null_rows():
    from pipeline.bronze_to_silver import clean_weather

    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "temperature": [-50.0, 10.0, None],
        "humidity": [50.0, 150.0, None],
        "solar_rad": [10.0, 20.0, None],
    })

    out = clean_weather(df)

    assert len(out) == 2
    assert out.iloc[0]["temperature"] == -30.0  # clamped from -50
    assert out.iloc[1]["humidity"] == 100.0  # clamped from 150


@patch("pipeline.bronze_to_silver.execute_values")
@patch("pipeline.bronze_to_silver.get_connection")
def test_bronze_to_silver_run_inserts_cleaned_rows(mock_get_connection, mock_execute_values):
    from pipeline.bronze_to_silver import run

    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    energy_rows = [{"timestamp": t, "kwh": 100.0 + i} for i, t in enumerate(ts)]
    weather_rows = [
        {"timestamp": t, "temperature": 10.0 + i, "humidity": 50.0, "solar_rad": 100.0}
        for i, t in enumerate(ts)
    ]
    mock_get_connection.return_value = _mock_conn_with_rows(energy_rows, weather_rows)
    mock_execute_values.side_effect = (
        lambda cur, sql, rows, **kw: [{"id": i} for i in range(len(rows))]
    )

    assert run() is None

    assert mock_execute_values.call_count == 2
    energy_call, weather_call = mock_execute_values.call_args_list
    assert "INSERT INTO clean_energy" in energy_call.args[1]
    assert "INSERT INTO clean_weather" in weather_call.args[1]
    assert len(energy_call.args[2]) == 3
    assert len(weather_call.args[2]) == 3


# ---------------------------------------------------------------------------
# silver_to_gold
# ---------------------------------------------------------------------------

def test_build_features_joins_and_computes_features():
    from pipeline.silver_to_gold import FEATURE_COLUMNS, build_features

    ts = pd.date_range("2024-07-31", periods=30, freq="h", tz="UTC")
    energy = pd.DataFrame({"timestamp": ts, "kwh": [100.0 + i for i in range(30)]})
    weather = pd.DataFrame({
        "timestamp": ts,
        "temperature": [15.0] * 30,
        "humidity": [50.0] * 30,
        "solar_rad": [0.0] * 30,
    })

    feat = build_features(energy, weather)

    assert len(feat) == 30
    assert list(feat.columns) == FEATURE_COLUMNS

    # 2024-08-01 is Nationalfeiertag (CH/ZH holiday). Zürich is UTC+2 in summer,
    # so local midnight Aug 1 falls at 22:00 UTC on Jul 31.
    holiday_row = feat[feat["timestamp"] == pd.Timestamp("2024-07-31 22:00", tz="UTC")].iloc[0]
    assert bool(holiday_row["is_holiday"]) is True
    assert bool(feat.iloc[0]["is_holiday"]) is False

    # lag_24h for row 24 (24 hours after row 0) should equal row 0's kwh.
    assert feat.iloc[24]["lag_24h"] == feat.iloc[0]["kwh"]
    assert pd.isna(feat.iloc[0]["lag_1h"])
    assert pd.isna(feat.iloc[0]["lag_24h"])


def test_build_features_empty_inputs_returns_empty():
    from pipeline.silver_to_gold import FEATURE_COLUMNS, build_features

    out = build_features(pd.DataFrame(), pd.DataFrame())

    assert out.empty
    assert list(out.columns) == FEATURE_COLUMNS


def test_build_features_left_join_keeps_unmatched_energy_rows():
    from pipeline.silver_to_gold import build_features

    ts = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    energy = pd.DataFrame({"timestamp": ts, "kwh": [100.0, 110.0, 120.0, 130.0, 140.0]})
    # Only one weather reading on record, matching just the 3rd energy timestamp
    # (mirrors the real MeteoSwiss source, which currently has a single row).
    weather = pd.DataFrame({
        "timestamp": [ts[2]],
        "temperature": [12.0],
        "humidity": [55.0],
        "solar_rad": [80.0],
    })

    feat = build_features(energy, weather)

    assert len(feat) == 5  # every energy row is kept, not just the matched one
    matched = feat[feat["timestamp"] == ts[2]].iloc[0]
    assert matched["temperature"] == 12.0
    unmatched = feat[feat["timestamp"] != ts[2]]
    assert unmatched["temperature"].isna().all()
    assert unmatched["humidity"].isna().all()
    assert unmatched["solar_rad"].isna().all()


def test_build_features_with_no_weather_data_at_all():
    from pipeline.silver_to_gold import build_features

    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    energy = pd.DataFrame({"timestamp": ts, "kwh": [100.0, 110.0, 120.0]})

    feat = build_features(energy, pd.DataFrame())

    assert len(feat) == 3
    assert feat["temperature"].isna().all()


@patch("pipeline.silver_to_gold.execute_values")
@patch("pipeline.silver_to_gold.get_connection")
def test_silver_to_gold_run_upserts_features(mock_get_connection, mock_execute_values):
    from pipeline.silver_to_gold import run

    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    energy_rows = [{"timestamp": t, "kwh": 100.0 + i} for i, t in enumerate(ts)]
    weather_rows = [
        {"timestamp": t, "temperature": 10.0, "humidity": 50.0, "solar_rad": 0.0} for t in ts
    ]
    # 3rd fetchall serves insert_features' existing-timestamps pre-check (1 row already exists).
    mock_get_connection.return_value = _mock_conn_with_rows(
        energy_rows, weather_rows, [{"timestamp": ts[0]}]
    )

    assert run() is None

    mock_execute_values.assert_called_once()
    _, sql, rows = mock_execute_values.call_args.args
    assert "ON CONFLICT (timestamp) DO UPDATE SET" in sql
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@patch("pipeline.validate.get_connection")
def test_validate_bronze_passes_under_threshold(mock_get_connection):
    from pipeline.validate import validate_bronze

    ts = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    rows = [{"timestamp": t, "kwh": (None if i == 0 else 100.0)} for i, t in enumerate(ts)]
    mock_get_connection.return_value = _mock_conn_with_rows(rows)

    assert validate_bronze() is True


@patch("pipeline.validate.get_connection")
def test_validate_bronze_fails_over_threshold(mock_get_connection):
    from pipeline.validate import validate_bronze

    ts = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    # 20% null kwh, above the 5% threshold.
    rows = [{"timestamp": t, "kwh": (None if i < 2 else 100.0)} for i, t in enumerate(ts)]
    mock_get_connection.return_value = _mock_conn_with_rows(rows)

    assert validate_bronze() is False


@patch("pipeline.validate.get_connection")
def test_validate_silver_detects_negative_kwh_and_duplicates(mock_get_connection):
    from pipeline.validate import validate_silver

    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    energy_rows = [
        {"timestamp": ts[0], "kwh": -5.0},
        {"timestamp": ts[1], "kwh": 100.0},
        {"timestamp": ts[1], "kwh": 110.0},  # duplicate timestamp
    ]
    weather_rows = [{"timestamp": ts[0]}, {"timestamp": ts[1]}]
    mock_get_connection.return_value = _mock_conn_with_rows(energy_rows, weather_rows)

    assert validate_silver() is False


@patch("pipeline.validate.get_connection")
def test_validate_silver_passes_clean_data(mock_get_connection):
    from pipeline.validate import validate_silver

    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    energy_rows = [{"timestamp": t, "kwh": 100.0} for t in ts]
    weather_rows = [{"timestamp": t} for t in ts]
    mock_get_connection.return_value = _mock_conn_with_rows(energy_rows, weather_rows)

    assert validate_silver() is True


@patch("pipeline.validate.get_connection")
def test_validate_gold_detects_missing_columns(mock_get_connection):
    from pipeline.validate import validate_gold

    rows = [{"timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "kwh": 100.0}]
    mock_get_connection.return_value = _mock_conn_with_rows(rows)

    assert validate_gold() is False


@patch("pipeline.validate.get_connection")
def test_validate_gold_passes_complete_features(mock_get_connection):
    from pipeline.validate import GOLD_FEATURE_COLUMNS, validate_gold

    ts = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    rows = [{col: (t if col == "timestamp" else 0) for col in GOLD_FEATURE_COLUMNS} for t in ts]
    mock_get_connection.return_value = _mock_conn_with_rows(rows)

    assert validate_gold() is True


def _gold_rows(ts, weather_null_count: int):
    """Build feature rows where every column is populated except
    temperature/humidity/solar_rad, which are NULL for the first
    `weather_null_count` rows (simulating sparse weather coverage)."""
    from pipeline.validate import GOLD_FEATURE_COLUMNS, GOLD_WEATHER_COLUMNS

    rows = []
    for i, t in enumerate(ts):
        row = {col: (t if col == "timestamp" else 0) for col in GOLD_FEATURE_COLUMNS}
        if i < weather_null_count:
            for col in GOLD_WEATHER_COLUMNS:
                row[col] = None
        rows.append(row)
    return rows


@patch("pipeline.validate.get_connection")
def test_validate_gold_tolerates_sparse_weather_under_80_percent_null(mock_get_connection):
    from pipeline.validate import validate_gold

    ts = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    # 75% of rows missing weather -- under the relaxed 80% threshold.
    mock_get_connection.return_value = _mock_conn_with_rows(_gold_rows(ts, weather_null_count=150))

    assert validate_gold() is True


@patch("pipeline.validate.get_connection")
def test_validate_gold_fails_when_weather_null_rate_exceeds_80_percent(mock_get_connection):
    from pipeline.validate import validate_gold

    ts = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    # 85% of rows missing weather -- over the relaxed 80% threshold.
    mock_get_connection.return_value = _mock_conn_with_rows(_gold_rows(ts, weather_null_count=170))

    assert validate_gold() is False


@patch("pipeline.validate.get_connection")
def test_validate_gold_still_strict_on_energy_columns(mock_get_connection):
    from pipeline.validate import GOLD_FEATURE_COLUMNS, validate_gold

    ts = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    rows = [{col: (t if col == "timestamp" else 0) for col in GOLD_FEATURE_COLUMNS} for t in ts]
    for row in rows[:10]:  # 5% null kwh -- well above the strict 1% threshold
        row["kwh"] = None
    mock_get_connection.return_value = _mock_conn_with_rows(rows)

    assert validate_gold() is False
