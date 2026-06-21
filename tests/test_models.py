"""Tests for model training, evaluation, and prediction."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_expected_keys():
    from models.evaluate import compute_metrics

    y_true = [100.0, 200.0, 300.0, 400.0]
    y_pred = [105.0, 195.0, 290.0, 410.0]

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics) == {"rmse", "mae", "r2", "mape", "estimated_cost_saving_pct"}
    assert metrics["mae"] == 7.5
    assert metrics["r2"] > 0.99


def test_estimated_cost_saving_pct_boundaries():
    from models.evaluate import estimated_cost_saving_pct

    assert estimated_cost_saving_pct(0.0) == 15.0  # perfect forecast -> max saving
    assert estimated_cost_saving_pct(4.0) == 12.5  # halfway to the threshold -> midpoint
    assert estimated_cost_saving_pct(8.0) == 0.0  # at threshold -> no assumed saving
    assert estimated_cost_saving_pct(20.0) == 0.0  # well above threshold
    assert estimated_cost_saving_pct(float("nan")) == 0.0


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

def _synthetic_gold_df(periods: int, start: str = "2024-01-01") -> pd.DataFrame:
    """A self-consistent synthetic Gold `features` table for training tests."""
    ts = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    kwh = 100 + 20 * np.sin(2 * np.pi * np.arange(periods) / 24) + rng.normal(0, 2, periods)

    df = pd.DataFrame({"timestamp": ts, "kwh": kwh})
    df["hour"] = ts.hour
    df["day_of_week"] = ts.dayofweek
    df["month"] = ts.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_holiday"] = False
    df["lag_1h"] = df["kwh"].shift(1)
    df["lag_24h"] = df["kwh"].shift(24)
    df["rolling_avg_7d"] = df["kwh"].rolling(24 * 7, min_periods=1).mean()
    df["temperature"] = 15.0
    df.loc[df.index % 10 == 0, "temperature"] = np.nan  # sparse weather coverage
    return df


def _mock_conn_returning(rows) -> MagicMock:
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn


# ---------------------------------------------------------------------------
# train_xgboost
# ---------------------------------------------------------------------------

def test_xgboost_split_train_test_uses_30_day_cutoff():
    from models.train_xgboost import split_train_test

    ts = pd.date_range("2024-01-01", periods=24 * 45, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "kwh": range(len(ts))})

    train_df, test_df = split_train_test(df, test_window_days=30)

    assert train_df["timestamp"].max() <= ts.max() - pd.Timedelta(days=30)
    assert test_df["timestamp"].min() > ts.max() - pd.Timedelta(days=30)
    assert len(train_df) + len(test_df) == len(df)


@patch("models.train_xgboost.get_connection")
def test_train_xgboost_end_to_end(mock_get_connection, tmp_path, monkeypatch):
    """Real XGBoost fit + a real (sqlite-backed, local) MLflow tracking/registry."""
    from models.train_xgboost import MODEL_NAME, train

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    df = _synthetic_gold_df(periods=24 * 45)
    mock_get_connection.return_value = _mock_conn_returning(df.to_dict("records"))

    metrics = train()

    assert set(metrics) == {"rmse", "mae", "r2", "mape", "estimated_cost_saving_pct"}
    mock_get_connection.return_value.close.assert_called_once()

    from mlflow import MlflowClient
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    assert len(versions) == 1


@patch("models.train_xgboost.get_connection")
def test_train_xgboost_raises_when_features_empty(mock_get_connection, tmp_path, monkeypatch):
    from models.train_xgboost import train

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mock_get_connection.return_value = _mock_conn_returning([])

    try:
        train()
        assert False, "expected RuntimeError for an empty features table"
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# train_prophet
# ---------------------------------------------------------------------------
# NOTE: this sandbox doesn't have cmdstan installed, so a real Prophet object
# can't even be constructed here. These tests fake out Prophet itself (and
# mlflow.prophet.log_model, whose real implementation serializes genuine
# Prophet internals) to validate the surrounding orchestration: the DB read,
# the CH/ZH holiday dataframe, the 30-day split, metric computation, and
# that registration is requested under the right model name.

def test_prophet_split_train_test_uses_30_day_cutoff():
    from models.train_prophet import split_train_test

    ts = pd.date_range("2024-01-01", periods=24 * 45, freq="h")
    df = pd.DataFrame({"ds": ts, "y": range(len(ts))})

    train_df, test_df = split_train_test(df, test_window_days=30)

    assert train_df["ds"].max() <= ts.max() - pd.Timedelta(days=30)
    assert test_df["ds"].min() > ts.max() - pd.Timedelta(days=30)


class _FakeProphet:
    """Stands in for prophet.Prophet, which can't run without cmdstan here."""

    def __init__(self, holidays=None, **kwargs):
        self.holidays = holidays
        self.params = kwargs
        self._train_mean = None

    def fit(self, df):
        self._train_mean = df["y"].mean()
        return self

    def predict(self, future_df):
        return pd.DataFrame({"ds": future_df["ds"], "yhat": [self._train_mean] * len(future_df)})


@patch("models.train_prophet.mlflow.prophet.log_model")
@patch("models.train_prophet.Prophet", _FakeProphet)
@patch("models.train_prophet.get_connection")
def test_train_prophet_orchestration(mock_get_connection, mock_log_model, tmp_path, monkeypatch):
    from models.train_prophet import MODEL_NAME, train

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    ts = pd.date_range("2024-01-01", periods=24 * 45, freq="h", tz="UTC")
    rows = [{"timestamp": t, "kwh": 100.0 + (i % 24)} for i, t in enumerate(ts)]
    mock_get_connection.return_value = _mock_conn_returning(rows)

    metrics = train()

    assert set(metrics) == {"rmse", "mae", "r2", "mape", "estimated_cost_saving_pct"}
    mock_log_model.assert_called_once()
    assert mock_log_model.call_args.kwargs["registered_model_name"] == MODEL_NAME


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class _RecordingXGBModel:
    """Fake XGBoost model: next kwh = lag_1h + 1, so chaining is traceable."""

    def __init__(self):
        self.calls = []

    def predict(self, features):
        self.calls.append(features.copy())
        return [features["lag_1h"].iloc[0] + 1]


def test_predict_xgboost_chains_lag_1h_and_uses_real_history_for_lag_24h():
    from models.predict import _predict_xgboost

    ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    history = pd.DataFrame({
        "timestamp": ts,
        "kwh": range(48),  # kwh == hour index, so lag values are easy to predict
        "rolling_avg_7d": 50.0,
    })

    model = _RecordingXGBModel()
    forecast = _predict_xgboost(model, history, horizon_hours=24)

    # last actual kwh is 47 -> step1 predicts 48, then each step chains +1.
    assert list(forecast["predicted_kwh"]) == [float(v) for v in range(48, 72)]
    assert forecast["timestamp"].iloc[0] == ts[-1] + pd.Timedelta(hours=1)
    assert forecast["timestamp"].iloc[-1] == ts[-1] + pd.Timedelta(hours=24)

    # lag_24h must always come from real history, never a chained prediction.
    first_call_lag_24h = model.calls[0]["lag_24h"].iloc[0]
    assert first_call_lag_24h == 24  # ts[-1] + 1h - 24h = ts[24], kwh = 24
    last_call_lag_24h = model.calls[-1]["lag_24h"].iloc[0]
    assert last_call_lag_24h == 47  # ts[-1] + 24h - 24h = ts[-1], kwh = 47


def test_predict_prophet_uses_naive_timestamps_and_maps_yhat():
    from models.predict import _predict_prophet

    last_ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")

    class _FakePredictModel:
        def predict(self, future_df):
            assert future_df["ds"].dt.tz is None  # Prophet requires tz-naive ds
            return pd.DataFrame({"ds": future_df["ds"], "yhat": range(len(future_df))})

    forecast = _predict_prophet(_FakePredictModel(), last_ts, horizon_hours=24)

    assert len(forecast) == 24
    assert forecast["timestamp"].iloc[0] == last_ts + pd.Timedelta(hours=1)
    assert list(forecast["predicted_kwh"]) == list(range(24))


def test_insert_predictions_writes_expected_rows():
    from models.predict import _insert_predictions

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        "predicted_kwh": [10.0, 20.0],
    })
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("models.predict.execute_values") as mock_execute_values:
        _insert_predictions(mock_conn, df, "xgboost-energy-zurich", "3")

    mock_execute_values.assert_called_once()
    _, sql, rows = mock_execute_values.call_args.args
    assert "INSERT INTO predictions" in sql
    assert rows == [
        (df["timestamp"].iloc[0], "xgboost-energy-zurich", "3", 10.0),
        (df["timestamp"].iloc[1], "xgboost-energy-zurich", "3", 20.0),
    ]
    mock_conn.commit.assert_called_once()


@patch("models.predict.execute_values")
@patch("models.predict.get_connection")
@patch("models.predict._read_recent_features")
@patch("models.predict._load_champion_model")
def test_predict_end_to_end_with_xgboost_flavor(
    mock_load_champion, mock_read_history, mock_get_connection, mock_execute_values
):
    from models.predict import predict

    ts = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    history = pd.DataFrame({"timestamp": ts, "kwh": range(48), "rolling_avg_7d": 50.0})

    mock_load_champion.return_value = (
        "xgboost", _RecordingXGBModel(), "xgboost-energy-zurich", "1"
    )
    mock_read_history.return_value = history
    mock_conn = MagicMock()
    mock_get_connection.return_value = mock_conn

    result = predict()

    assert len(result) == 24
    assert set(result[0]) == {"timestamp", "predicted_kwh"}
    mock_execute_values.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("models.predict._load_champion_model")
def test_predict_returns_empty_list_when_no_model_available(mock_load_champion):
    from models.predict import predict

    mock_load_champion.side_effect = RuntimeError("no models registered")

    assert predict() == []


@patch("models.predict.get_connection")
@patch("models.predict._read_recent_features")
@patch("models.predict._load_champion_model")
def test_predict_returns_empty_list_when_no_history(
    mock_load_champion, mock_read_history, mock_get_connection
):
    from models.predict import predict

    mock_load_champion.return_value = ("xgboost", MagicMock(), "xgboost-energy-zurich", "1")
    mock_read_history.return_value = pd.DataFrame()

    assert predict() == []
    mock_get_connection.assert_not_called()
