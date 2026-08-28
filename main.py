import pandas as pd
import matplotlib.pyplot as plt

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
forecast = model_fit.forecast(steps=len(test))

# Step 5: Compare forecast vs actual 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"MAE  (avg error size):        {mae:.3f}")
print(f"RMSE (penalizes big misses):  {rmse:.3f}")

# save the numbers to a simple text file so we don't lose them
with open("results/metrics.txt", "w") as f:
    f.write("ARIMA(2,1,2) on ETTh2 - Oil Temperature\n")
    f.write(f"Train size: {len(train)} hours\n")
    f.write(f"Test size: {len(test)} hours\n")
    f.write(f"MAE:  {mae:.3f}\n")
    f.write(f"RMSE: {rmse:.3f}\n")

print("Saved metrics to results/metrics.txt")

# Step 6: Plot actual vs predicted so we can SEE how it did
plt.figure(figsize=(12, 5))
plt.plot(train.index[-100:], train.values[-100:], label="Train (last 100 hrs)")
plt.plot(test.index, test.values, label="Actual", color="green")
plt.plot(test.index, forecast, label="Forecast", color="red", linestyle="--")
plt.title("ARIMA Forecast vs Actual - ETTh2 Oil Temperature")
plt.xlabel("Date")
plt.ylabel("Oil Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("results/forecast_plot.png")
print("Saved forecast plot to results/forecast_plot.png")
