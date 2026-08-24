from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ForecastResult:
    model: Pipeline
    metrics: dict[str, float]
    test_frame: pd.DataFrame


def generate_dataset(n: int = 720, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic, non-sensitive campus demand dataset."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n, freq="D")
    weekday = days.dayofweek.to_numpy()
    exam_week = ((days.month.isin([3, 5, 10, 12])) & (days.day > 10)).astype(int)
    rain = rng.gamma(shape=1.4, scale=2.0, size=n)
    temperature = 25 + 7 * np.sin(np.arange(n) / 30) + rng.normal(0, 1.2, n)
    attendance = np.clip(0.78 - 0.08 * (weekday >= 5) - 0.018 * rain + rng.normal(0, .03, n), .2, .95)
    demand = (
        110
        + 22 * attendance
        + 13 * (weekday < 5)
        + 18 * exam_week
        + 2.2 * temperature
        - 1.8 * rain
        + rng.normal(0, 5, n)
    )
    return pd.DataFrame({
        "date": days,
        "weekday": weekday,
        "exam_week": exam_week,
        "rain_mm": rain.round(2),
        "temperature_c": temperature.round(2),
        "attendance_rate": attendance.round(3),
        "demand_units": demand.round(2),
    })


def train_model(frame: pd.DataFrame, seed: int = 42) -> ForecastResult:
    features = ["weekday", "exam_week", "rain_mm", "temperature_c", "attendance_rate"]
    X = frame[features]
    y = frame["demand_units"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)
    prep = ColumnTransformer([("numeric", "passthrough", features)], remainder="drop")
    model = Pipeline([
        ("features", prep),
        ("regressor", RandomForestRegressor(n_estimators=160, random_state=seed, min_samples_leaf=3)),
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(mean_squared_error(y_test, predicted) ** .5),
        "test_rows": float(len(X_test)),
    }
    test_frame = X_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["predicted"] = predicted
    return ForecastResult(model, metrics, test_frame)


def time_ordered_split(frame: pd.DataFrame, test_size: float = .2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-indexed frame chronologically for leakage-aware evaluation."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("frame must contain at least two rows")
    split = min(max(1, int(round(len(frame) * (1 - test_size)))), len(frame) - 1)
    ordered = frame.sort_values("date").reset_index(drop=True)
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def evaluate(seed: int = 42) -> dict[str, float]:
    return train_model(generate_dataset(seed=seed), seed=seed).metrics
