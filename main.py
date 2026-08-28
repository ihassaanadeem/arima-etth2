import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the data 
df = pd.read_csv("data/ETTh2.csv")

# make the date column an actual datetime and use it as the index
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

print("Data loaded. Shape:", df.shape)
print(df.head())

# we only care about Oil Temperature (OT) for this task
ot = df["OT"]

# quick plot just to LOOK at the data before doing anything fancy
plt.figure(figsize=(12, 4))
plt.plot(ot)
plt.title("ETTh2 - Oil Temperature over time")
plt.xlabel("Date")
plt.ylabel("Oil Temperature")
plt.tight_layout()
plt.savefig("results/raw_data_plot.png")
print("Saved raw data plot to results/raw_data_plot.png")

# Step 2: Train/test split 
# we are NOT using the full 17420 hours, that's way too slow for ARIMA on a laptop.
# instead we take a smaller recent chunk so this actually finishes running.
ot_subset = ot[-1000:]  # last 1000 hours (~41 days)

train_size = int(len(ot_subset) * 0.8)
train, test = ot_subset[:train_size], ot_subset[train_size:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# Step 3: Fit ARIMA on the training data 
from statsmodels.tsa.arima.model import ARIMA

# order = (p, d, q)
# p=2 -> look at last 2 values, d=1 -> difference once to remove trend,
# q=2 -> look at last 2 prediction errors
# these are just reasonable starting numbers, not "the" correct ones
order = (2, 1, 2)

model = ARIMA(train, order=order)
model_fit = model.fit()

print(model_fit.summary())

# Step 4: Forecast the test period 
# ask the model to predict as many steps ahead as we have test data for
arima_forecast = model_fit.forecast(steps=len(test))

# Step 5: Compare forecast vs actual 
from sklearn.metrics import mean_absolute_error, mean_squared_error

arima_mae = mean_absolute_error(test, arima_forecast)
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))

print(f"ARIMA MAE  (avg error size):        {arima_mae:.3f}")
print(f"ARIMA RMSE (penalizes big misses):  {arima_rmse:.3f}")

# Step 6: Random Forest on the same data 
from sklearn.ensemble import RandomForestRegressor

# random forest can't take a raw time series, so we turn past values into
# input columns (lag features) it can learn from
n_lags = 5


def make_lag_features(series, n_lags):
    feat_df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        feat_df[f"lag_{lag}"] = feat_df["y"].shift(lag)
    return feat_df.dropna()


feat_df = make_lag_features(ot_subset, n_lags)
X = feat_df.drop(columns="y")
y = feat_df["y"]

split_date = test.index[0]
X_train, X_test = X[X.index < split_date], X[X.index >= split_date]
y_train, y_test = y[y.index < split_date], y[y.index >= split_date]

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# forecast the test period step by step, feeding each prediction back in
# as the "latest known value" for the next step, same idea as ARIMA above
history = list(ot_subset[ot_subset.index < split_date].values[-n_lags:])
rf_forecast = []

lag_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)]

for i in range(len(test)):
    x_input = pd.DataFrame([history[-n_lags:][::-1]], columns=lag_cols)
    pred = rf_model.predict(x_input)[0]
    rf_forecast.append(pred)
    history.append(pred)

rf_forecast = pd.Series(rf_forecast, index=test.index)

rf_mae = mean_absolute_error(y_test, rf_forecast)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_forecast))

print(f"Random Forest MAE:  {rf_mae:.3f}")
print(f"Random Forest RMSE: {rf_rmse:.3f}")

# Step 7: save the numbers to a simple text file so we don't lose them
with open("results/metrics.txt", "w") as f:
    f.write("ETTh2 Oil Temperature - ARIMA vs Random Forest\n")
    f.write(f"Train size: {len(train)} hours\n")
    f.write(f"Test size: {len(test)} hours\n\n")
    f.write("ARIMA(2,1,2)\n")
    f.write(f"MAE:  {arima_mae:.3f}\n")
    f.write(f"RMSE: {arima_rmse:.3f}\n\n")
    f.write(f"Random Forest ({n_lags} lag features)\n")
    f.write(f"MAE:  {rf_mae:.3f}\n")
    f.write(f"RMSE: {rf_rmse:.3f}\n")

print("Saved metrics to results/metrics.txt")

# Step 8: Plot actual vs both forecasts so we can SEE how they did
plt.figure(figsize=(12, 5))
plt.plot(train.index[-100:], train.values[-100:], label="Train (last 100 hrs)")
plt.plot(test.index, test.values, label="Actual", color="green")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast", color="red", linestyle="--")
plt.plot(test.index, rf_forecast, label="Random Forest Forecast", color="orange", linestyle="--")
plt.title("ARIMA vs Random Forest - ETTh2 Oil Temperature")
plt.xlabel("Date")
plt.ylabel("Oil Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("results/forecast_plot.png")
print("Saved forecast plot to results/forecast_plot.png")