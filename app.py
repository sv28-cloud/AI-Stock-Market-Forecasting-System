import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="AI Stock Forecasting Dashboard", layout="wide")

st.title("📈 AI Stock Market Forecasting")
st.markdown("Compare **LSTM vs ARIMA** and visualize prediction signals")

# =====================================================
# Sidebar controls
# =====================================================
ticker = st.sidebar.text_input("Enter stock ticker:", "^GSPC")
start_date = st.sidebar.date_input("Start date:", pd.to_datetime("2010-01-01"))
seq_length = st.sidebar.slider("LSTM Sequence Length (days)", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 5, 30, 15)

if st.sidebar.button("Run Forecasting"):

    # --- 1. Load Data ---
    df = yf.download(ticker, start=start_date)
    df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]]

    # Create Tomorrow column and Target in one step to avoid alignment issues
    df["Tomorrow"] = df["Close"].shift(-1)
    
    # Create Target column before dropping NaN to ensure alignment
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)

    # --- 2. Feature Engineering ---
    horizons = [2, 5, 60, 250, 1000]
    predictors = []
    
    for horizon in horizons:
        if len(df) > horizon:
            rolling_averages = df.rolling(horizon, min_periods=1).mean()
            
            ratio_column = f"Close_Ratio_{horizon}"
            df[ratio_column] = df["Close"] / (rolling_averages["Close"] + 1e-8)

            trend_column = f"Trend_{horizon}"
            df[trend_column] = df.shift(1).rolling(horizon, min_periods=1).sum()["Target"]

            predictors += [ratio_column, trend_column]

    df = df.dropna().reset_index(drop=True)
    
    if len(df) < seq_length + 100:
        st.error(f"Not enough data for analysis. Need at least {seq_length + 100} rows, got {len(df)}")
        st.stop()

    # =====================================================
    # 🔹 LSTM MODEL
    # =====================================================
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[predictors])

    def create_sequences(data, targets, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(targets[i+seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_features, df["Target"].values, seq_length)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # =====================================================
    # 🔹 Train model (adaptive validation)
    # =====================================================
    with st.spinner("Training LSTM model..."):
        if epochs <= 5:  # quick runs
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
        else:  # longer runs → include validation
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                verbose=1
            )

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    lstm_accuracy = accuracy_score(y_test, y_pred)
    lstm_precision = precision_score(y_test, y_pred, zero_division=0)

    lstm_start_idx = seq_length + split
    lstm_end_idx = lstm_start_idx + len(y_pred)
    lstm_indices = range(lstm_start_idx, lstm_end_idx)
    lstm_predictions = pd.Series(y_pred, index=lstm_indices, name="LSTM_Pred")

    # =====================================================
    # 🔹 ARIMA BASELINE
    # =====================================================
    split_idx = int(0.8 * len(df))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    with st.spinner("Training ARIMA model..."):
        try:
            model_arima = ARIMA(train["Close"], order=(5,1,0))
            arima_fit = model_arima.fit()
            forecast = arima_fit.forecast(steps=len(test))
            
            test = test.copy()
            test["Pred_Close"] = forecast.values
            test["Pred_Target"] = (test["Pred_Close"].shift(-1) > test["Pred_Close"]).astype(int)
            test = test.dropna()

            if len(test) > 0:
                arima_accuracy = accuracy_score(test["Target"], test["Pred_Target"])
                arima_precision = precision_score(test["Target"], test["Pred_Target"], zero_division=0)
            else:
                arima_accuracy = 0.0
                arima_precision = 0.0
                
        except Exception as e:
            st.warning(f"ARIMA model failed: {str(e)}")
            arima_accuracy = 0.0
            arima_precision = 0.0
            test = df.iloc[split_idx:].copy()
            test["Pred_Close"] = test["Close"]

    # =====================================================
    # 🔹 Results Display
    # =====================================================
    st.subheader("📊 Model Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LSTM Accuracy", f"{lstm_accuracy:.4f}")
        st.metric("LSTM Precision", f"{lstm_precision:.4f}")
    
    with col2:
        st.metric("ARIMA Accuracy", f"{arima_accuracy:.4f}")
        st.metric("ARIMA Precision", f"{arima_precision:.4f}")
    
    improvement = (lstm_accuracy - arima_accuracy) * 100
    if improvement > 0:
        st.success(f"✅ LSTM improvement over ARIMA: {improvement:.2f}%")
    else:
        st.info(f"📊 ARIMA vs LSTM difference: {improvement:.2f}%")

    # =====================================================
    # 🔹 Plot training history (Accuracy)
    # =====================================================
    st.subheader("📉 LSTM Training Performance (Accuracy)")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history.history["accuracy"], label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history.history:
        ax.plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)

    ax.set_title("LSTM Training History", fontsize=16)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # =====================================================
    # 🔹 Plot training history (Loss)
    # =====================================================
    st.subheader("📉 LSTM Training Performance (Loss)")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

    ax_loss.plot(history.history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history.history:
        ax_loss.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)

    ax_loss.set_title("LSTM Loss History", fontsize=16)
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    st.pyplot(fig_loss)

    # =====================================================
    # 🔹 Actual vs ARIMA forecast
    # =====================================================
    st.subheader("📈 Actual vs ARIMA Forecasted Close Prices")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(test.index, test["Close"], label="Actual Close", linewidth=2, color='blue')
    ax2.plot(test.index, test["Pred_Close"], label="ARIMA Forecast", linewidth=2, color='orange')
    ax2.set_title(f"ARIMA Forecast vs Actual for {ticker}", fontsize=16)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # =====================================================
    # 🔹 LSTM Predicted Signals Visualization
    # =====================================================
    st.subheader("🚦 LSTM Buy/Sell Signals vs Actual Close Price")
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    ax3.plot(df.index, df["Close"], label="Close Price", color="blue", alpha=0.7, linewidth=1)

    valid_indices = [idx for idx in lstm_predictions.index if idx < len(df)]
    if valid_indices:
        valid_predictions = lstm_predictions.loc[valid_indices]
        
        buy_signals = valid_predictions[valid_predictions == 1]
        sell_signals = valid_predictions[valid_predictions == 0]
        
        if len(buy_signals) > 0:
            ax3.scatter(buy_signals.index, df.loc[buy_signals.index, "Close"], 
                       marker="^", color="green", s=50, label="LSTM Predict Up", alpha=0.8)
        
        if len(sell_signals) > 0:
            ax3.scatter(sell_signals.index, df.loc[sell_signals.index, "Close"], 
                       marker="v", color="red", s=50, label="LSTM Predict Down", alpha=0.8)

    ax3.set_title(f"LSTM Buy/Sell Signals for {ticker}", fontsize=16)
    ax3.set_xlabel("Time Index")
    ax3.set_ylabel("Price")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    
    # =====================================================
    # 🔹 Summary statistics
    # =====================================================
    st.subheader("📋 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Points", len(df))
        st.metric("Training Split", f"{split_idx} rows")
    
    with col2:
        if valid_indices:
            buy_count = len(lstm_predictions[lstm_predictions == 1])
            sell_count = len(lstm_predictions[lstm_predictions == 0])
            st.metric("LSTM Buy Signals", buy_count)
            st.metric("LSTM Sell Signals", sell_count)
    
    with col3:
        st.metric("Features Used", len(predictors))
        st.metric("Sequence Length", seq_length)
