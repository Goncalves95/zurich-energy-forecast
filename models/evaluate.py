"""Evaluate model performance: RMSE, MAE, R² and business metric translation."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Business assumption: a forecast accurate enough to hit MAPE < 8% lets
# procurement buy closer to actual peak demand, worth an estimated 10-15%
# saving on peak energy costs. We linearly interpolate within that band
# (MAPE near 8% -> ~10% saving, MAPE near 0% -> ~15% saving); at/above 8%
# MAPE there's no assumed saving.
MAPE_SAVING_THRESHOLD_PCT = 8.0
MIN_SAVING_PCT = 10.0
MAX_SAVING_PCT = 15.0


def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%), ignoring entries where y_true is 0 (percentage error is undefined)."""
    nonzero = y_true != 0
    if not nonzero.any():
        return float("nan")
    errors = np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
    return float(np.mean(errors) * 100)


def estimated_cost_saving_pct(mape: float) -> float:
    """Translate forecast accuracy (MAPE, in %) into an estimated peak
    procurement cost saving, per the 10-15% business assumption at MAPE < 8%."""
    if not np.isfinite(mape) or mape >= MAPE_SAVING_THRESHOLD_PCT:
        return 0.0
    accuracy_ratio = 1 - (mape / MAPE_SAVING_THRESHOLD_PCT)
    saving = MIN_SAVING_PCT + accuracy_ratio * (MAX_SAVING_PCT - MIN_SAVING_PCT)
    return round(saving, 2)


def compute_metrics(y_true, y_pred) -> dict:
    """Compute RMSE, MAE, R², MAPE, and the derived cost-saving estimate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mape = _mean_absolute_percentage_error(y_true, y_pred)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
        "estimated_cost_saving_pct": estimated_cost_saving_pct(mape),
    }
