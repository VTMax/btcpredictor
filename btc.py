import csv
import sklearn.metrics
import time
from datetime import datetime
import requests
import pytz
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np
base_url = 'https://api.coingecko.com/api/v3'
api_key = 'INSERT_API_KEY_HERE'

def get_bitcoin_price():
    url = f'{base_url}/simple/price?ids=bitcoin&vs_currencies=usd'
    headers = {
        'Accepts': 'application/json',
        'X-CoinGecko-API-Key': api_key
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['bitcoin']['usd']
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def generate_features(timestamps):
    features = np.array([timestamps, np.sin(timestamps), np.cos(timestamps)]).T
    return features.reshape(-1, features.shape[-1])

def train_ridge_model(X, y):
    model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=0.1))
    model.fit(X, y)
    return model

def train_random_forest_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_gradient_boosting_model(X, y):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    return model

def calculate_accuracy(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def make_predictions(models, timestamps):
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(generate_features(np.array(timestamps).reshape(-1, 1)))
    return predictions

def fetch_historical_data():
    timestamps, prices = [], []
    for _ in range(10):
        current_price = get_bitcoin_price()
        if current_price is not None:
            current_timestamp_utc = datetime.now(pytz.utc)
            current_timestamp_pst = current_timestamp_utc.astimezone(pytz.timezone('America/Los_Angeles'))
            timestamps.append(current_timestamp_pst.timestamp())
            prices.append(current_price)
            time.sleep(10)
    return timestamps, prices

def main():
    timestamps, prices = fetch_historical_data()

    X = generate_features(np.array(timestamps).reshape(-1, 1))
    y = np.array(prices)

    ridge_model = train_ridge_model(X, y)
    rf_model = train_random_forest_model(X, y)
    gb_model = train_gradient_boosting_model(X, y)

    models = {'Ridge': ridge_model, 'RandomForest': rf_model, 'GradientBoosting': gb_model}

    future_timestamps = [timestamps[-1] + 60 * i for i in range(1, 1441)]

    predictions = make_predictions(models, future_timestamps)

    predictions_file_path = 'combined_predictions.csv'
    with open(predictions_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Ridge_Prediction', 'RandomForest_Prediction', 'GradientBoosting_Prediction'])

        for timestamp, rf_pred, lr_pred, gd_pred in zip(future_timestamps, predictions['Ridge'], predictions['RandomForest'], predictions['GradientBoosting']):
            timestamp_str = datetime.fromtimestamp(timestamp, pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d %I:%M:%S %p %Z')
            writer.writerow([timestamp_str, float(rf_pred), float(lr_pred), float(gd_pred)])

    combined_data = pd.read_csv(predictions_file_path)


if __name__ == "__main__":
    main()
"""
This module fetches historical Bitcoin price data, trains multiple machine learning models on the data, makes predictions for future prices using the models, and outputs the predictions to a CSV file.

The main functions are:

- fetch_historical_data: Gets recent Bitcoin prices from CoinGecko API
- generate_features: Creates feature vectors from timestamp data 
- train_MODEL_model: Trains Ridge, Random Forest, and Gradient Boosting models
- make_predictions: Generates predictions from trained models
- main: Runs data fetching, model training, prediction, and CSV output
"""