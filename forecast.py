from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
import logging

# Configure Prophet logger
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def prepare_time_series_data(df, crime_type=None, state=None):
    """Prepare time series data with optional filters"""
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    elif 'Timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        raise ValueError("No valid datetime column found")

    # Apply filters if provided
    if crime_type and 'Crime Head' in df.columns:
        df = df[df['Crime Head'] == crime_type]
    if state and 'State/UT' in df.columns:
        df = df[df['State/UT'] == state]

    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.normalize()  # Remove time component
    return df.groupby('date').size().reset_index(name='count').rename(columns={'date': 'ds', 'count': 'y'})

def train_forecaster(df, seasonality_mode='additive', changepoint_prior_scale=0.05):
    """Train Prophet model with configurable parameters"""
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale
    )
    model.fit(df)
    return model

def make_forecast(model, periods=30, freq='D'):
    """Generate forecast with evaluation metrics"""
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    
    # Calculate MAE if we have actual values
    if len(model.history) > periods:
        actual = model.history['y'][-periods:]
        predicted = forecast['yhat'][-periods:]
        mae = mean_absolute_error(actual, predicted)
        forecast['mae'] = mae
    
    return forecast

def plot_forecast(model, forecast):
    """Create interactive Plotly forecast visualization"""
    fig = model.plot(forecast)
    fig = go.Figure(fig)
    fig.update_layout(
        title='Crime Forecast',
        xaxis_title='Date',
        yaxis_title='Crime Count',
        hovermode='x unified'
    )
    return fig