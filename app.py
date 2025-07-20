import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import plotly.graph_objs as go
import shap
import yfinance as yf
import numpy as np

st.title("📈 SmartQuant: Stock Movement Predictor")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")

# Load processed data and model for AAPL
@st.cache_resource
def load_aapl_data_and_model():
    df = pd.read_csv('AAPL_2019_2025_processed.csv', index_col=0)
    target_col = 'Close'
    features = [col for col in df.columns if col != target_col]
    split_idx = int(0.8 * len(df))
    train = df.iloc[:split_idx]
    X_train, y_train = train[features], train[target_col]
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    return df, model, features, explainer

df, model, features, explainer = load_aapl_data_and_model()

# Helper: fetch and preprocess new ticker (demo: use last 60 days, fill missing columns with 0)
def fetch_and_preprocess_ticker(ticker, features):
    try:
        data = yf.download(ticker, period="3mo")
        if data.empty or len(data) < 30:
            return None, "Not enough data for this ticker."
        # Basic preprocessing: keep only required columns, fill missing with 0
        data = data.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        # Create dummy columns for missing features
        for col in features:
            if col not in data.columns:
                data[col] = 0
        # Use only the last row for prediction
        latest = data.iloc[[-1]][features]
        return latest, None
    except Exception as e:
        return None, str(e)

# Plotly chart
if st.checkbox("Show Price Chart"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='AAPL Close Price'))
    st.plotly_chart(fig)

# SHAP feature importance
if st.checkbox("Show Feature Importance (SHAP)"):
    shap_values = explainer.shap_values(df[features].iloc[-100:])
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({'feature': features, 'importance': shap_sum})
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    st.bar_chart(importance_df.set_index('feature'))

if st.button("Predict"):
    if ticker.upper() == "AAPL":
        latest = df.iloc[[-1]][features]
        pred = model.predict(latest)[0]
        last_close = df.iloc[-1]['Close']
        direction = "📈 UP" if pred > last_close else "📉 DOWN"
        st.success(f"Prediction for {ticker.upper()}: {direction} (Predicted Close: {pred:.2f})")
    else:
        latest, err = fetch_and_preprocess_ticker(ticker, features)
        if latest is None:
            st.error(f"Could not fetch or preprocess data for {ticker.upper()}: {err}")
        else:
            pred = model.predict(latest)[0]
            st.success(f"Prediction for {ticker.upper()}: Predicted Close: {pred:.2f}") 