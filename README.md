# 📈 Tesla Stock Price Prediction using LSTM (Next 7-Day Forecast)

This project is a **time series forecasting model** that predicts the **next 7 days of Tesla (TSLA) stock prices** based on the past 60 days of historical data. It uses a **deep learning model (LSTM - Long Short-Term Memory)** trained on 15 years of historical stock data to predict key financial indicators like Open, High, Low, Close, and Volume.

---

## 🎯 Project Goals

- Predict short-term Tesla stock prices (7-day horizon) using deep learning
- Understand trends in TSLA stock using historical time series data
- Visualize predictions alongside actual stock trends
- Evaluate performance using Mean Absolute Error (MAE)

---

## 🔁 Workflow

1. **Data Collection**
   - Uses `yfinance` to fetch 15 years of Tesla stock data.
   - Data includes Open, High, Low, Close, Volume, Dividends, and Stock Splits.

2. **Data Preprocessing**
   - Drops non-essential columns (`Dividends`, `Stock Splits`).
   - Converts date column to datetime and sets it as index.
   - Applies `MinMaxScaler` to normalize values between 0 and 1.

3. **Sequence Building**
   - For each point in time, uses the **previous 60 days of data** to predict the **next 7 days**.
   - Each sample contains features for Open, High, Low, Close, and Volume.

4. **Model Architecture**
   - LSTM (128 units) + Dropout (0.2) to prevent overfitting
   - Dense layer with 35 units (7 days × 5 features)
   - Loss function: Mean Squared Error
   - Optimizer: Adam

5. **Training & Validation**
   - 80/20 train-test split (preserving temporal order)
   - Trains over 50 epochs with batch size 32
   - Loss and validation loss plotted for monitoring

6. **Prediction**
   - Predicts the next 7 days from the most recent 60 days
   - Inverse transformation to get actual price values
   - Visualizes predicted vs actual close prices

7. **Evaluation**
   - Calculates **Mean Absolute Error (MAE)** for Close price prediction

---

## 📊 Techniques Used

| Area                 | Technique / Tool                         |
|----------------------|-------------------------------------------|
| Data Collection      | `yfinance`                                |
| Preprocessing        | `pandas`, `MinMaxScaler` from `sklearn`   |
| Deep Learning Model  | `LSTM`, `Dropout`, `Dense` (`TensorFlow`) |
| Evaluation Metric    | Mean Absolute Error (MAE)                 |
| Visualization        | `matplotlib`, `seaborn`                   |

---

## 🧪 Requirements

Install dependencies via pip:
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow
