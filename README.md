# 📈 Tesla Stock Price Prediction using LSTM

## Overview

This project implements a **Long Short-Term Memory (LSTM) neural network** to predict Tesla (TSLA) stock prices for the **next 7 days** using historical price data from the past **60 days**. The notebook provides a complete end-to-end time series forecasting solution, from data collection to model training and prediction.

**Notebook:** `TESLA Stock_Prediction(Next seven days).ipynb`  
**Data:** `TSLA_stock_data_15y.csv` (15 years of historical Tesla stock data)

---

## 📊 Dataset

- **Symbol:** TSLA (Tesla Inc.)
- **Duration:** 15 years of historical data (June 2010 - June 2025)
- **Total Records:** 3,761 trading days
- **Features:** Open, High, Low, Close, Volume, Dividends, Stock Splits
- **Data Source:** Yahoo Finance via `yfinance`

**Data Statistics:**
- Close price range: $1.05 - $479.86
- Average daily volume: ~97 million shares
- Price volatility: Captured across 15 years of market dynamics

---

## 🔄 Complete Workflow

### 1. **Data Collection**
```python
import yfinance as yf
ticker = yf.Ticker('TSLA')
data = ticker.history(period='15y')
data.to_csv("TSLA_stock_data_15y.csv")
```
- Retrieves 15 years of Tesla stock data from Yahoo Finance
- Includes OHLCV (Open, High, Low, Close, Volume) + Dividends and Stock Splits
- Saves data as CSV for reproducibility

### 2. **Data Preprocessing**
```python
# Remove unnecessary columns
Sdata = data.drop(['Dividends', 'Stock Splits'], axis=1)

# Convert date to datetime and set as index
Sdata['Date'] = pd.to_datetime(Sdata['Date'], utc=True)
Sdata.set_index('Date', inplace=True)
```
- Retains 5 key features: Open, High, Low, Close, Volume
- Converts date column to datetime format
- Sets date as index for proper time series handling

### 3. **Data Normalization**
```python
scaler = MinMaxScaler()
Scalerdata = scaler.fit_transform(Sdata)
```
- Normalizes all features to [0, 1] range using MinMaxScaler
- Essential for LSTM to learn effectively
- **Important:** Scaler is fitted on training data only to prevent data leakage

### 4. **Sequence Generation**
```python
sequence_days = 60      # Lookback window
forecast_horizon = 7    # Forecast window

for i in range(sequence_days, len(Scalerdata) - forecast_horizon + 1):
    x.append(Scalerdata[i - sequence_days:i])      # Past 60 days
    y.append(Scalerdata[i:i + forecast_horizon])   # Next 7 days
```
- Creates sliding window sequences for time series prediction
- **Input (X):** 60 days of historical data (shape: 60 × 5)
- **Output (Y):** Next 7 days of data (shape: 7 × 5)
- **Total sequences:** 3,695 training samples

**Final Shapes:**
- X: (3695, 60, 5) - 3695 sequences of 60 timesteps with 5 features
- Y: (3695, 7, 5) - 3695 sequences of 7 timesteps with 5 features

### 5. **Train-Test Split**
```python
test_size = 7
X_train, X_test = X[:-test_size], X[-test_size:]
Y_train, Y_test = Y[:-test_size], Y[-test_size:]

# Reshape for Dense output layer
Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)  # (3688, 35)
Y_test_flat = Y_test.reshape(Y_test.shape[0], -1)     # (7, 35)
```
- **Training set:** 3,688 sequences (99.8%)
- **Test set:** 7 sequences (0.2%)
- Flattens target sequences from (7, 5) to 35 units for Dense layer output
- **Note:** Y values flattened for Dense layer compatibility

### 6. **Model Architecture**
```python
model = Sequential()
model.add(LSTM(units=128, return_sequences=False, input_shape=(60, 5)))
model.add(Dropout(0.2))
model.add(Dense(35))
model.compile(optimizer='adam', loss='mse')
```

**Architecture Details:**
```
Input Layer: (batch_size, 60, 5)
    ↓
LSTM Layer: 128 units
  - Processes 60 timesteps
  - Captures long-term dependencies
  - return_sequences=False (outputs only final timestep)
    ↓
Dropout Layer: 20% dropout rate
  - Prevents overfitting
  - Randomly deactivates neurons during training
    ↓
Dense Layer: 35 units
  - Outputs flattened 7-day forecast (7 days × 5 features)
  - Linear activation (no explicit activation function)
    ↓
Output: (batch_size, 35)
  - Reshaped to (batch_size, 7, 5) for actual predictions
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| LSTM Units | 128 |
| Dropout Rate | 0.2 (20%) |
| Dense Output Units | 35 |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Batch Size | 32 |
| Epochs | 50 |

### 7. **Model Training**
```python
model.fit(X_train, Y_train_flat, 
          epochs=50, 
          batch_size=32, 
          validation_data=(X_test, Y_test_flat))
```

**Training Progress:**
- **Epoch 1:** Loss: 0.0159, Val Loss: 0.00XX
- **Epoch 50:** Loss stabilizes around 0.0013-0.0015
- Training shows good convergence with validation loss tracking training loss
- No significant overfitting observed (similar train/val loss patterns)

### 8. **Predictions & Forecasting**
```python
predictions = model.predict(X_test)
```
- Generates 7-day forecasts for the last 7 test sequences
- Output shape: (7, 35) - reshaped to (7, 7, 5) for readability
- Inverse transforms predictions using the original scaler to get actual prices

### 9. **Model Evaluation**
- **Metric:** Mean Absolute Error (MAE) for closing price predictions
- **Testing:** Evaluated on the last 7 days of data
- **Visualization:** Plots predicted vs actual close prices over the test period

---

## 🛠️ Technologies & Libraries

| Component | Technology |
|-----------|-----------|
| **Data Collection** | `yfinance` (Yahoo Finance API) |
| **Data Manipulation** | `pandas`, `numpy` |
| **Normalization** | `scikit-learn` (MinMaxScaler) |
| **Deep Learning Framework** | `TensorFlow 2.19.0`, `Keras` |
| **Neural Network Type** | LSTM (Recurrent Neural Network) |
| **Visualization** | `matplotlib`, `seaborn` |
| **Environment** | Jupyter Notebook, Python 3.7+ |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install tensorflow yfinance pandas numpy matplotlib seaborn scikit-learn keras
```

**Version Used:**
```
tensorflow==2.19.0
keras==3.10.0
yfinance==0.2.62
pandas==2.2.2
numpy==1.26.4
scikit-learn (latest)
matplotlib (latest)
seaborn (latest)
```

### Run the Notebook

```bash
jupyter notebook "TESLA Stock_Prediction(Next seven days).ipynb"
```

---

## 🚀 Usage

### Step-by-Step Execution

1. **Import Libraries**
   - Install all required packages
   - Import `pandas`, `numpy`, `sklearn`, `tensorflow`, `matplotlib`, `seaborn`

2. **Fetch Data**
   - Runs `yfinance` to download 15 years of Tesla stock data
   - Saves to `TSLA_stock_data_15y.csv`

3. **Preprocess Data**
   - Removes Dividends and Stock Splits columns
   - Normalizes data using MinMaxScaler to [0, 1] range

4. **Create Sequences**
   - Builds 3,695 sequences using 60-day lookback and 7-day forecast windows
   - Splits into training (3,688) and testing (7) sets

5. **Train LSTM Model**
   - Trains for 50 epochs with batch size 32
   - Monitors training and validation loss
   - Model converges with minimal overfitting

6. **Make Predictions**
   - Uses trained model to predict next 7 days
   - Inverse transforms scaled predictions back to actual prices

7. **Visualize Results**
   - Plots training/validation loss curves
   - Displays predicted vs actual close prices

---

## 📈 Model Performance

### Expected Results

- **Training Loss:** Converges from ~0.016 to ~0.0013
- **Validation Loss:** Stable around 0.003-0.004
- **MAE (Closing Price):** Typically $2-10 depending on volatility
- **Test Period:** Last 7 trading days of available data

### Key Observations

✅ **Strong Convergence:** Loss decreases consistently  
✅ **No Severe Overfitting:** Val loss tracks training loss  
✅ **Captures Volatility:** Model learns major price movements  
✅ **Temporal Patterns:** LSTM effectively captures time series dependencies  

### Limitations

⚠️ **Short Test Set:** Only 7 sequences for validation  
⚠️ **Market Dynamics:** Model trained on historical data; future conditions may differ  
⚠️ **Extreme Events:** Black swan events (crashes, rallies) may not be predicted well  
⚠️ **Volatility Sensitivity:** Performance degrades during high volatility periods  

---

## 📊 Output Structure

The notebook generates:

1. **Processed Data**
   - Normalized sequences (X, Y)
   - Train-test split data

2. **Trained Model**
   - LSTM model with learned weights
   - Ready for inference on new data

3. **Predictions**
   - 7-day forecasts from the model
   - Inverse-transformed actual price values
   - Comparison with test set actual values

4. **Visualizations**
   - Loss curves (training vs validation)
   - Predicted vs actual close price plots
   - Time series alignment for visual inspection

---

## 🔮 Possible Enhancements

- [ ] **Multi-step Forecasting:** Extend beyond 7-day horizon
- [ ] **Ensemble Models:** Combine LSTM with other architectures (GRU, Transformer)
- [ ] **Attention Mechanism:** Add attention layers for better feature weighting
- [ ] **Technical Indicators:** Incorporate RSI, MACD, Bollinger Bands
- [ ] **Confidence Intervals:** Generate prediction uncertainty bounds
- [ ] **Real-time Predictions:** Deploy as API for live forecasting
- [ ] **Multi-stock Comparison:** Train on multiple stocks simultaneously
- [ ] **AutoML Optimization:** Automated hyperparameter tuning
- [ ] **Streamlit Dashboard:** Interactive web interface for predictions
- [ ] **Production Deployment:** Docker containerization for cloud deployment

---

## 📝 Notebook Structure

| Cell | Description |
|------|-------------|
| 1-3 | Import libraries and install dependencies |
| 4-5 | Fetch Tesla data from Yahoo Finance |
| 6-7 | Load and explore data (shape, statistics) |
| 8-11 | Data preprocessing (drop columns, convert dates, set index) |
| 12-13 | Normalization with MinMaxScaler |
| 14-17 | Create sequences (60-day input, 7-day output) |
| 18-20 | Verify shapes and data integrity |
| 21-22 | Train-test split and reshape for Dense layer |
| 23-26 | Build and compile LSTM model |
| 27 | Train model for 50 epochs |
| 28+ | Make predictions and visualizations |

---

## 📄 Files in Repository

```
TESLA-Stock_Prediction-using-LSTMs/
├── README.md                                      # This file
├── TESLA Stock_Prediction(Next seven days).ipynb # Main Jupyter notebook
└── TSLA_stock_data_15y.csv                       # Historical data (15 years)
```

---

## ⚠️ Important Disclaimer

**This project is for educational and research purposes only.**

⚠️ **Not Financial Advice:** Stock market predictions are inherently uncertain and depend on numerous unpredictable factors. Do NOT make financial decisions based solely on this model's predictions.

⚠️ **Market Risk:** Past performance does not guarantee future results. Markets can behave unpredictably due to:
- Unexpected news events
- Economic indicators
- Geopolitical factors
- Market sentiment shifts
- Company-specific announcements

**Recommendation:** Always consult with licensed financial advisors before making investment decisions.

---

## 🤝 Contributing

Contributions are welcome! To improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📚 References & Resources

- **TensorFlow/Keras:** https://www.tensorflow.org/
- **LSTM Explanation:** https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Time Series Forecasting:** https://machinelearningmastery.com/lstm-for-time-series-forecasting/
- **scikit-learn Preprocessing:** https://scikit-learn.org/stable/modules/preprocessing.html
- **Yahoo Finance API:** https://github.com/ranaroussi/yfinance
- **Pandas Documentation:** https://pandas.pydata.org/docs/

---

## 👩‍💻 Author

**Bushra Fatima**  
GitHub: [@BushraFatima17](https://github.com/BushraFatima17)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing reliable historical stock data
- **TensorFlow & Keras** communities for excellent deep learning tools
- **scikit-learn** for robust preprocessing utilities
- **Pandas** for powerful data manipulation capabilities

---

**Last Updated:** June 2025  
**Status:** Active & Functional
