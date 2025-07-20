import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler

# Load data, skip header and metadata rows, set correct header, parse dates
raw = pd.read_csv('AAPL_2019_2025.csv', skiprows=3, header=None, names=['Date','Close','High','Low','Open','Volume'])
raw['Date'] = pd.to_datetime(raw['Date'])
raw.set_index('Date', inplace=True)

# Convert columns to numeric
for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
    raw[col] = pd.to_numeric(raw[col], errors='coerce')

# Handle missing values (forward fill, then drop any remaining)
raw = raw.ffill().dropna()

# Add all TA features
raw = add_all_ta_features(
    raw, open="Open", high="High", low="Low", close="Close", volume="Volume"
)

# Drop original price/volume columns if not needed
raw = raw.drop(['Open', 'High', 'Low', 'Volume'], axis=1)

# Drop columns with more than 30% missing values
thresh = int(0.7 * len(raw))
raw = raw.dropna(axis=1, thresh=thresh)

# Remove any remaining NaNs (rows)
raw = raw.dropna()

# Normalize features
features_to_normalize = [col for col in raw.columns if col != 'Close']
scaler = MinMaxScaler()
raw_norm = raw.copy()
raw_norm[features_to_normalize] = scaler.fit_transform(raw[features_to_normalize])

# Save processed data
raw_norm.to_csv('AAPL_2019_2025_processed.csv')

print('Feature engineering complete. Saved as AAPL_2019_2025_processed.csv') 