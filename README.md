# ETTh2 Oil Temperature Forecasting with ARIMA

This is a small internship task where I apply ARIMA (a classic statistical
forecasting method) on the ETTh2 dataset to predict Oil Temperature (OT).

## What is ETTh2?
ETTh2 stands for "Electricity Transformer Temperature, hourly, station 2".
It's a public dataset of hourly readings from a power transformer in China,
originally used to benchmark time series forecasting models (including
Transformer-based ones). More info / original source:
https://github.com/zhouhaoyi/ETDataset

## What I did
1. Loaded the ETTh2 dataset and looked at the Oil Temperature (OT) column.
2. Took the last 1000 hours of data (full dataset is slow for ARIMA to run on).
3. Split it into 800 hours train / 200 hours test.
4. Fit an ARIMA(2,1,2) model on the training data.
5. Forecasted the next 200 hours and compared to the real values.
6. Measured error (MAE, RMSE) and plotted actual vs predicted.

## Files
- `main.py` - the full script, run top to bottom
- `data/ETTh2.csv` - the dataset
- `results/raw_data_plot.png` - what the raw data looks like
- `results/forecast_plot.png` - predicted vs actual comparison
- `results/metrics.txt` - error scores

## How to run
```
pip install -r requirements.txt
python main.py
```

## What I found
The model tracks the real data reasonably well for the first day or so of
the forecast, but then flattens out toward a roughly constant value while
the actual oil temperature keeps oscillating. This is a known limitation of
ARIMA on data with a lot of short-term noise/cycles - it's better suited for
short-horizon forecasts than long ones. MAE and RMSE values are in
`results/metrics.txt`.

## Next steps (not done yet)
- Try different (p,d,q) values or use auto_arima to search for better ones
- Try forecasting fewer steps ahead to see if short-term accuracy improves
- Compare against a simple baseline (e.g. "tomorrow = today")
