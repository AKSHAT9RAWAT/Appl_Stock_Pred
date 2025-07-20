# SmartQuant: Stock Movement Predictor

A web app for stock price movement prediction using technical indicators, Decision Tree model, and Streamlit. Supports feature importance visualization (SHAP), price charting (Plotly), and predictions for any ticker (demo).

## Features

- Download and preprocess stock data (AAPL by default)
- Feature engineering with technical indicators (`ta` library)
- Train/test split and Decision Tree regression
- SHAP feature importance visualization
- Interactive Streamlit app with:
  - Ticker input
  - Price chart (Plotly)
  - Feature importance bar chart
  - Prediction for next close (AAPL model, demo for other tickers)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/AKSHAT9RAWAT/Appl_Stock_Pred
   cd smartquant-stock-predictor
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

## Usage

1. Preprocess data and train model:
   ```bash
   python preprocess_stock_data.py
   python train_model.py
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes

- The model is trained on AAPL data. Predictions for other tickers are for demo only.
- For real predictions on other stocks, retrain the model on their data.

