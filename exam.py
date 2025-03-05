import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Set up Streamlit app
st.title(" Stock Price Prediction Web App")
st.sidebar.header("Stock Selection")

# User input for stock selection
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*15))
end_date = st.sidebar.date_input("End Date", datetime.now())

# User input for forecast period
forecast_years = st.sidebar.slider("Select Forecast Period (Years)", min_value=1, max_value=5, value=1, step=1)
forecast_period = forecast_years * 365  # Convert years to days

# Fetch stock data
@st.cache_data
def get_stock_data(symbol, start, end):
    stock = yf.download(symbol, start=start, end=end, progress=False)
    stock = stock.dropna()
    return stock

df = get_stock_data(stock_symbol, start_date, end_date)

if df.empty:
    st.error(" No stock data found! Please enter a valid stock symbol.")
    st.stop()

# Display stock data
st.subheader(f"Stock Data for {stock_symbol}")
st.write(df)

# Prophet Forecasting
st.subheader(" Prophet Model Forecast")

# Prepare data for Prophet
df_prophet = df[['Close']].reset_index()
df_prophet.columns = ['ds', 'y']

# Train Prophet model
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# Create future dates
future_dates = prophet_model.make_future_dataframe(periods=forecast_period)
forecast = prophet_model.predict(future_dates)

# Display Prophet Forecasted Prices
st.write("**Prophet Forecasted Prices (Future First):**")
st.dataframe(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Prophet Forecast'}).iloc[::-1])

# Plot Prophet results
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Predicted Price"))
fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name="Actual Price"))
fig2.update_layout(title="Prophet Prediction", xaxis_title="Date", yaxis_title="Stock Price (USD)")
st.plotly_chart(fig2)

# ARIMA Forecasting
st.subheader(" ARIMA Forecasting")

# Function for ARIMA forecast
def arima_forecast(data, period):
    train = data['Close'].dropna()
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=period)
    forecast_dates = pd.date_range(data.index[-1], periods=period+1, freq='D')[1:]
    return pd.DataFrame({'Date': forecast_dates, 'ARIMA Forecast': forecast.values}), model_fit

# Train ARIMA model & get forecast
arima_forecast_df, arima_model = arima_forecast(df, forecast_period)

# Display ARIMA Forecasted Prices
st.write("**ARIMA Forecasted Prices (Future First):**")
st.dataframe(arima_forecast_df.iloc[::-1])

# ARIMA Plot
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label="Actual", color='blue')
ax.plot(arima_forecast_df["Date"], arima_forecast_df["ARIMA Forecast"], label="ARIMA Forecast", color='orange')
ax.set_title("ARIMA Forecast")
ax.legend()
st.pyplot(fig)

# Model Performance Evaluation
st.subheader(" Model Performance Evaluation")

# Splitting dataset for training (80%)
train_size = int(len(df) * 0.8)
train_arima = df['Close'][:train_size]
train_prophet = df_prophet.iloc[:train_size]

# Function to evaluate model performance
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(true, predicted) * 100  # Convert to percentage
    accuracy = 100 - mape  # Accuracy formula
    return mae, mse, rmse, mape, accuracy

# Get ARIMA predictions on training set
train_predictions_arima = arima_model.predict(start=0, end=train_size - 1)

# Evaluate ARIMA model
mae_arima, mse_arima, rmse_arima, mape_arima, accuracy_arima = evaluate_model(train_arima, train_predictions_arima)

# Evaluate Prophet model on historical data
prophet_pred_train = forecast.iloc[:train_size]['yhat']
prophet_actual_train = train_prophet['y']

mae_prophet, mse_prophet, rmse_prophet, mape_prophet, accuracy_prophet = evaluate_model(prophet_actual_train, prophet_pred_train)

# Display Model Performance Metrics
st.write("###  ARIMA Model Performance")
st.write(f"**MAE:** {mae_arima:.2f}")
st.write(f"**MSE:** {mse_arima:.2f}")
st.write(f"**RMSE:** {rmse_arima:.2f}")
st.write(f"**MAPE:** {mape_arima:.2f}%")
st.write(f"**Accuracy:** {accuracy_arima:.2f}%")

st.write("###  Prophet Model Performance")
st.write(f"**MAE:** {mae_prophet:.2f}")
st.write(f"**MSE:** {mse_prophet:.2f}")
st.write(f"**RMSE:** {rmse_prophet:.2f}")
st.write(f"**MAPE:** {mape_prophet:.2f}%")
st.write(f"**Accuracy:** {accuracy_prophet:.2f}%")

# Ensemble Forecast (Prophet + ARIMA)
st.subheader(" Ensemble Forecast (Prophet + ARIMA)")

# Merge ARIMA & Prophet results
forecast_combined = forecast[['ds', 'yhat']].copy()
forecast_combined['arima'] = np.nan

# Assign ARIMA values correctly
forecast_combined.loc[forecast_combined.index[-forecast_period:], 'arima'] = arima_forecast_df['ARIMA Forecast'].values
forecast_combined['ensemble'] = forecast_combined[['yhat', 'arima']].mean(axis=1)

# Display Ensemble Forecasted Prices
st.write("**Ensemble Forecasted Prices (Future First):**")
st.dataframe(forecast_combined[['ds', 'ensemble']].rename(columns={'ds': 'Date', 'ensemble': 'Ensemble Forecast'}).iloc[::-1])

# Plot Ensemble Model
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=forecast_combined['ds'], y=forecast_combined['yhat'], mode='lines', name="Prophet Prediction"))
fig4.add_trace(go.Scatter(x=forecast_combined['ds'], y=forecast_combined['arima'], mode='lines', name="ARIMA Prediction"))
fig4.add_trace(go.Scatter(x=forecast_combined['ds'], y=forecast_combined['ensemble'], mode='lines', name="Ensemble Prediction"))
fig4.update_layout(title="Ensemble Prediction", xaxis_title="Date", yaxis_title="Stock Price (USD)")
st.plotly_chart(fig4)

# Ensemble Model Performance Evaluation
st.subheader(" Ensemble Model Performance")

# Find common dates between actual stock data and predictions
common_dates = df.index.intersection(forecast_combined['ds'])

# Extract matching actual values and predicted values
ensemble_actual = df.loc[common_dates, 'Close']
ensemble_pred = forecast_combined.loc[forecast_combined['ds'].isin(common_dates), 'ensemble']

# Ensure index alignment
ensemble_pred.index = ensemble_actual.index  # Align indices explicitly

# Evaluate Ensemble Model Performance
mae_ensemble, mse_ensemble, rmse_ensemble, mape_ensemble, accuracy_ensemble = evaluate_model(ensemble_actual, ensemble_pred)


# Display Ensemble Model Performance
st.write(f" **MAE:** {mae_ensemble:.2f}")
st.write(f" **MSE:** {mse_ensemble:.2f}")
st.write(f" **RMSE:** {rmse_ensemble:.2f}")
st.write(f" **MAPE:** {mape_ensemble:.2f}%")
st.write(f" **Accuracy:** {accuracy_ensemble:.2f}%")

st.success(" Thank you for using the Stock Price Prediction Web App!")
