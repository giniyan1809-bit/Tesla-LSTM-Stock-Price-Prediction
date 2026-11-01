import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Tesla Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Tesla Stock Price Prediction Using LSTM")
st.write(
    "This educational project uses historical Tesla (TSLA) closing prices and an "
    "LSTM neural network to forecast a next-day price estimate."
)
st.warning("This is an educational forecast, not financial advice. Stock prices are uncertain; do not use this result alone to make investment decisions.")

with st.sidebar:
    st.header("Project settings")
    period = st.selectbox("Historical data period", ["2y", "5y", "10y"], index=1)
    window = st.slider("Past trading days used per prediction", min_value=20, max_value=120, value=60, step=10)
    epochs = st.slider("Training epochs", min_value=5, max_value=50, value=15, step=5)

@st.cache_data(ttl=3600)
def load_data(period_choice):
    data = yf.download("TSLA", period=period_choice, interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    # yfinance may return a MultiIndex column layout in some versions.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Close"]].dropna().copy()
    data.index = pd.to_datetime(data.index)
    return data

@st.cache_resource(show_spinner=False)
def train_model(close_values, dates_values, window_size, epochs_count):
    values = np.asarray(close_values, dtype=np.float32).reshape(-1, 1)
    split_index = int(len(values) * 0.8)
    if split_index <= window_size or len(values) - split_index < 2:
        raise ValueError("Not enough rows for the selected sequence window. Choose a longer historical period.")

    # Fit scaler on training data only to reduce data leakage.
    scaler = MinMaxScaler()
    scaler.fit(values[:split_index])
    scaled = scaler.transform(values)

    X, y, target_indices = [], [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])
        target_indices.append(i)
    X = np.asarray(X, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.asarray(y, dtype=np.float32)
    target_indices = np.asarray(target_indices)

    train_mask = target_indices < split_index
    test_mask = target_indices >= split_index
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    if len(X_test) == 0:
        raise ValueError("Not enough test data. Choose a longer historical period.")

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=epochs_count,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        shuffle=False,
        verbose=0
    )
    pred_scaled = model.predict(X_test, verbose=0)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred_actual = scaler.inverse_transform(pred_scaled).ravel()

    next_sequence = scaled[-window_size:].reshape(1, window_size, 1)
    next_scaled = model.predict(next_sequence, verbose=0)
    next_price = float(scaler.inverse_transform(next_scaled)[0, 0])

    test_dates = pd.to_datetime(np.asarray(dates_values)[target_indices[test_mask]])
    return model, scaler, y_test_actual, pred_actual, test_dates, next_price

data = load_data(period)
if data.empty:
    st.error("Could not download TSLA data. Check your internet connection and try again.")
    st.stop()

st.subheader("Historical Tesla closing prices")
st.line_chart(data["Close"])
st.caption(f"Dataset contains {len(data):,} daily observations from {data.index.min().date()} to {data.index.max().date()}.")

data_index = data.index
if st.button("Train LSTM and predict", type="primary"):
    try:
        with st.spinner("Training the LSTM model... this may take a little while on the first run."):
            model, scaler, actual, predicted, dates, next_price = train_model(
                tuple(data["Close"].astype(float).tolist()), tuple(data.index.astype(str)), window, epochs
            )
        st.success("Model training and evaluation completed.")
        c1, c2, c3 = st.columns(3)
        mae = mean_absolute_error(actual, predicted)
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        c1.metric("MAE", f"${mae:,.2f}")
        c2.metric("RMSE", f"${rmse:,.2f}")
        c3.metric("Next trading-day estimate", f"${next_price:,.2f}")

        results = pd.DataFrame({"Actual Close": actual, "Predicted Close": predicted}, index=dates)
        st.subheader("Actual vs predicted prices (test data)")
        st.line_chart(results)
        st.subheader("Prediction results")
        st.dataframe(results.tail(15).style.format("${:,.2f}"), use_container_width=True)
        st.download_button(
            "Download test predictions as CSV",
            data=results.to_csv(index_label="Date").encode("utf-8"),
            file_name="tesla_lstm_predictions.csv",
            mime="text/csv"
        )
        st.info("MAE and RMSE are measured in dollars; lower values indicate smaller errors on this test period. Past performance does not guarantee future results.")
    except Exception as exc:
        st.error(f"Model could not be trained: {exc}")
        st.caption("Try selecting a longer historical period or fewer epochs.")
else:
    st.info("Choose settings in the sidebar, then click **Train LSTM and predict**.")
