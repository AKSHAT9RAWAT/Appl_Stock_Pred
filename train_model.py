import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv('AAPL_2019_2025_processed.csv', index_col=0)

target_col = 'Close'
features = [col for col in df.columns if col != target_col]

# Time-series train-test split (no shuffling)
split_idx = int(0.8 * len(df))
train, test = df.iloc[:split_idx], df.iloc[split_idx:]
X_train, y_train = train[features], train[target_col]
X_test, y_test = test[features], test[target_col]

# Option 1: Decision Tree
def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f'Decision Tree Test MSE: {mse:.6f}')
    # SHAP interpretability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print('SHAP summary plot saved as shap_summary.png')
    return model

# Option 2: LightGBM
def train_lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f'LightGBM Test MSE: {mse:.6f}')
    return model

if __name__ == '__main__':
    print('Training Decision Tree model...')
    train_decision_tree(X_train, y_train, X_test, y_test)
    # To use LightGBM instead, uncomment below:
    # print('Training LightGBM model...')
    # train_lightgbm(X_train, y_train, X_test, y_test) 