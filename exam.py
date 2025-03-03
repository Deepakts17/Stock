import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Set up Streamlit app
st.title("Stock Price Prediction Web App")
st.sidebar.header("Stock Selection")

# User input for stock selection
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*15))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Fetch stock data
@st.cache_data
def get_stock_data(symbol, start, end):
    stock = yf.download(symbol, start=start, end=end, progress=False)
    stock = stock.dropna()
    return stock

df = get_stock_data(stock_symbol, start_date, end_date)

if df.empty:
    st.error("⚠️ No stock data found! Please enter a valid stock symbol.")
    st.stop()

# Display stock data
st.subheader(f"Stock Data for {stock_symbol}")
st.write(df)

# Plot historical stock prices
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Closing Price"))
# fig.update_layout(title="Stock Price Over Time", xaxis_title="Date", yaxis_title="Stock Price (USD)")
# st.plotly_chart(fig)

# Forecasting with Prophet
st.subheader("Prophet Model Forecast")

# Prepare data for Prophet
df_prophet = df[['Close']].reset_index()
df_prophet.columns = ['ds', 'y']

# Train Prophet model
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# Create future dates
future_dates = prophet_model.make_future_dataframe(periods=365)
forecast = prophet_model.predict(future_dates)

# Display Prophet Forecasted Prices in Reverse Order
st.write("📌 **Prophet Forecasted Prices (Future First):**")
st.dataframe(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Prophet Forecast'}).iloc[::-1])

# Plot Prophet results
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Predicted Price"))
fig2.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name="Actual Price"))
fig2.update_layout(title="Prophet Prediction", xaxis_title="Date", yaxis_title="Stock Price (USD)")
st.plotly_chart(fig2)

# Forecasting with ARIMA
def arima_forecast(data, period):
    train = data['Close'].dropna()
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=period)
    forecast_dates = pd.date_range(data.index[-1], periods=period+1, freq='D')[1:]
    return pd.DataFrame({'Date': forecast_dates, 'ARIMA Forecast': forecast.values})

st.subheader("ARIMA Forecasting")
period = 365
arima_forecast_df = arima_forecast(df, period)

# Display Forecasted Prices in Reverse Order
st.write("📌 **ARIMA Forecasted Prices (Future First):**")
st.dataframe(arima_forecast_df.iloc[::-1])

# ARIMA Plot
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label="Actual", color='blue')
ax.plot(arima_forecast_df["Date"], arima_forecast_df["ARIMA Forecast"], label="ARIMA Forecast", color='orange')
ax.set_title("ARIMA Forecast")
ax.legend()
st.pyplot(fig)

# Combined Ensemble Model (Average of Prophet & ARIMA)
st.subheader("Ensemble Forecast (Prophet + ARIMA)")

# Merge ARIMA & Prophet results
forecast_combined = forecast[['ds', 'yhat']].copy()
forecast_combined['arima'] = np.nan

# Assign ARIMA values correctly
forecast_combined.loc[forecast_combined.index[-365:], 'arima'] = arima_forecast_df['ARIMA Forecast'].values
forecast_combined['ensemble'] = forecast_combined[['yhat', 'arima']].mean(axis=1)

# Display Ensemble Forecasted Prices in Reverse Order
st.write("📌 **Ensemble Forecasted Prices (Future First):**")
st.dataframe(forecast_combined[['ds', 'ensemble']].rename(columns={'ds': 'Date', 'ensemble': 'Ensemble Forecast'}).iloc[::-1])

# Plot Ensemble Model
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=forecast_combined['ds'], y=forecast_combined['yhat'], mode='lines', name="Prophet Prediction"))
fig4.add_trace(go.Scatter(x=forecast_combined['ds'], y=forecast_combined['arima'], mode='lines', name="ARIMA Prediction"))
fig4.add_trace(go.Scatter(x=forecast_combined['ds'], y=forecast_combined['ensemble'], mode='lines', name="Ensemble Prediction"))
fig4.update_layout(title="Ensemble Prediction", xaxis_title="Date", yaxis_title="Stock Price (USD)")
st.plotly_chart(fig4)

st.success("Thank you for using Stock Price Prediction Web App")

