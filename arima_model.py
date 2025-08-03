import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def forecast_arima(df: pd.DataFrame, steps: int, interval: str = "1d", column: str = "Close") -> pd.Series:
    """
    Fit an ARIMA model and forecast future prices aligned with real-time progression.
    """
    series = df[column].dropna()

    if len(series) < 20:
        raise ValueError("At least 20 data points are required for ARIMA.")

    # Use the actual existing datetime index
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Index must be a datetime index.")

    # Map frequency strings
    freq_map = {"1h": "H", "1d": "D", "1wk": "W", "1mo": "M"}
    freq = freq_map.get(interval, "D")

    # Fit ARIMA on the real series (DO NOT reset index)
    model = ARIMA(series, order=(5, 1, 0))
    fitted_model = model.fit()

    # Forecast future steps
    forecast = fitted_model.forecast(steps=steps)

    # Build real future datetime index
    last_date = series.index[-1]
    future_index = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq), periods=steps, freq=freq)
    forecast.index = future_index
    forecast.name = "Forecast"
    print(f"forecast shape: {forecast.shape}, index: {forecast.index}")
    return forecast
