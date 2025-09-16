# cfo_dashboard/backend/services/forecasting_service.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
from typing import Optional

# Ignore warnings from ARIMA for non-stationary data
warnings.filterwarnings("ignore")

# Try to import Prophet, fallback gracefully if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

def _generate_forecast_arima(data: pd.Series, forecast_steps: int = 12) -> pd.DataFrame:
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame(alpha=0.05)
    forecast_df = forecast_df.rename(columns={
        'mean': 'forecast',
        'mean_ci_lower': 'lower_bound',
        'mean_ci_upper': 'upper_bound'
    })
    return forecast_df[['forecast', 'lower_bound', 'upper_bound']]


def _prepare_supervised_from_series(series: pd.Series, num_lags: int = 12) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in range(1, num_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    df.dropna(inplace=True)
    return df


def _generate_forecast_ml(series: pd.Series, model_name: str, forecast_steps: int = 12) -> pd.DataFrame:
    df = _prepare_supervised_from_series(series, num_lags=min(12, max(3, len(series)//6)))
    feature_cols = [c for c in df.columns if c.startswith("lag_")]
    X, y = df[feature_cols].values, df["y"].values

    if model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)

    model.fit(X, y)

    # Recursive forecasting
    last_window = df.iloc[-1][feature_cols].values.astype(float)
    preds = []
    for _ in range(forecast_steps):
        next_val = float(model.predict(last_window.reshape(1, -1))[0])
        preds.append(next_val)
        # update window: drop oldest lag, prepend new value
        last_window = np.roll(last_window, 1)
        last_window[0] = next_val

    # Build forecast frame with naive uncertainty bounds based on train residuals
    residuals = y - model.predict(X)
    std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

    future_index = pd.date_range(start=series.index[-1] + (series.index.freq or pd.infer_freq(series.index) or pd.offsets.MonthBegin(1)), periods=forecast_steps, freq=series.index.freq or pd.infer_freq(series.index) or 'MS')
    forecast_df = pd.DataFrame({
        'forecast': preds,
        'lower_bound': np.array(preds) - 1.96 * std,
        'upper_bound': np.array(preds) + 1.96 * std,
    }, index=future_index)
    return forecast_df


def _generate_forecast_prophet(data: pd.Series, forecast_steps: int = 12) -> pd.DataFrame:
    """Generate forecast using Facebook Prophet."""
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not available. Please install it with: pip install prophet")
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    df = pd.DataFrame({
        'ds': data.index,
        'y': data.values
    })
    
    # Initialize and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add custom seasonality if we have enough data
    if len(data) >= 24:  # At least 2 years of monthly data
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=4)
    
    model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_steps, freq='MS')
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Extract only the forecast period
    forecast_period = forecast.tail(forecast_steps)
    
    # Convert to expected format
    result_df = pd.DataFrame({
        'forecast': forecast_period['yhat'].values,
        'lower_bound': forecast_period['yhat_lower'].values,
        'upper_bound': forecast_period['yhat_upper'].values
    }, index=forecast_period['ds'])
    
    return result_df


def generate_forecast(data: pd.Series, forecast_steps: int = 12, model: str = "arima") -> pd.DataFrame:
    """Generate forecast using ARIMA, Prophet, or ML regressors.

    model: one of ["arima", "prophet", "random_forest", "gradient_boosting"]
    """
    if model == "arima":
        return _generate_forecast_arima(data, forecast_steps)
    elif model == "prophet":
        return _generate_forecast_prophet(data, forecast_steps)
    elif model in ("random_forest", "gradient_boosting"):
        return _generate_forecast_ml(data, model, forecast_steps)
    else:
        return _generate_forecast_arima(data, forecast_steps)


def generate_multiple_forecasts(data: pd.Series, forecast_steps: int = 12) -> dict:
    """Generate forecasts using multiple models for comparison."""
    forecasts = {}
    
    # ARIMA forecast
    try:
        forecasts['arima'] = _generate_forecast_arima(data, forecast_steps)
    except Exception as e:
        forecasts['arima'] = None
        print(f"ARIMA forecast failed: {e}")
    
    # Prophet forecast (if available)
    if PROPHET_AVAILABLE:
        try:
            forecasts['prophet'] = _generate_forecast_prophet(data, forecast_steps)
        except Exception as e:
            forecasts['prophet'] = None
            print(f"Prophet forecast failed: {e}")
    
    # Random Forest forecast
    try:
        forecasts['random_forest'] = _generate_forecast_ml(data, "random_forest", forecast_steps)
    except Exception as e:
        forecasts['random_forest'] = None
        print(f"Random Forest forecast failed: {e}")
    
    # Gradient Boosting forecast
    try:
        forecasts['gradient_boosting'] = _generate_forecast_ml(data, "gradient_boosting", forecast_steps)
    except Exception as e:
        forecasts['gradient_boosting'] = None
        print(f"Gradient Boosting forecast failed: {e}")
    
    return forecasts


def calculate_forecast_accuracy(actual: pd.Series, forecast: pd.Series) -> dict:
    """Calculate forecast accuracy metrics."""
    if len(actual) != len(forecast):
        min_len = min(len(actual), len(forecast))
        actual = actual.iloc[:min_len]
        forecast = forecast.iloc[:min_len]
    
    mae = np.mean(np.abs(actual - forecast))
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2)
    }