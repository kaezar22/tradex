import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------- Robust Yahoo Finance Fetch ----------------------


def fetch_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         interval=interval, progress=False)

        if df.empty:
            return df

        # Fix column format issues
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.columns.tolist() == [ticker] * 5:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        st.warning(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# ---------------------- Chop Zone Indicator ----------------------


def Chop_Zone(df: pd.DataFrame, longitud: int = 30, longitud_ema: int = 34, columna: str = "Close") -> pd.Series:
    TP = (df["High"] + df["Low"] + df["Close"]) / 3
    precio_suavizado = df[columna].rolling(
        window=longitud, min_periods=longitud)
    max_suavizado = precio_suavizado.max()
    min_suavizado = precio_suavizado.min()
    rango_HL = 25 / (max_suavizado - min_suavizado) * min_suavizado

    ema = df[columna].ewm(
        span=longitud_ema, min_periods=longitud_ema, adjust=False).mean()
    y2_ema = (ema.shift(1) - ema) / TP * rango_HL
    c_ema = np.sqrt(1 + y2_ema ** 2)
    angulo_ema0 = round(np.rad2deg(np.arccos(1 / c_ema)))
    angulo_ema1 = np.where(y2_ema > 0, -angulo_ema0,
                           angulo_ema0)[max(longitud, longitud_ema):]
    CZ = np.where(angulo_ema1 >= 5, 0,
                  np.where((angulo_ema1 >= 3.57), 1,
                           np.where((angulo_ema1 >= 2.14), 2,
                           np.where((angulo_ema1 >= 0.71), 3,
                                    np.where(angulo_ema1 <= -5, 4,
                                    np.where((angulo_ema1 <= -3.57), 5,
                                             np.where((angulo_ema1 <= -2.14), 6,
                                             np.where((angulo_ema1 <= -0.71), 7, 8))))))))

    return pd.Series([np.nan] * max(longitud, longitud_ema) + CZ.tolist(), index=df.index, name="CZ")


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Chop Zone Screener", layout="wide")
st.title("📈 Chop Zone Market Screener")

tickers_input = st.text_area(
    "Enter up to 30 Yahoo Finance tickers (separated by commas)",
    "AAPL, MSFT, TSLA, NVDA, AMZN"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

interval = st.selectbox("Select Time Interval", ["1h", "1d", "1wk", "1mo"])

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime(2025, 7, 3))

run_button = st.button("Run Evaluation")

# ---------------------- Main Logic ----------------------
if run_button:
    if not tickers:
        st.error("❌ Please enter at least one valid ticker.")
    else:
        results = []
        cz_matrix = {}

        with st.spinner("📊 Downloading data and evaluating..."):
            for ticker in tickers:
                df = fetch_data(ticker, str(start_date),
                                str(end_date), interval)

                if df.empty or len(df) < 50:
                    st.warning(f"⚠️ Not enough data for {ticker}, skipping.")
                    continue

                cz = Chop_Zone(df).dropna()
                if cz.empty:
                    st.warning(
                        f"⚠️ Chop Zone output empty for {ticker}, skipping.")
                    continue

                last_score = cz.iloc[-1]
                results.append({"Ticker": ticker, "Score": last_score})

                series = df["Close"].loc[cz.index]
                if isinstance(series, pd.Series) and not series.empty:
                    cz_matrix[ticker] = series

        if not results or len(cz_matrix) == 0:
            st.error(
                "❌ No valid instruments processed. Check the date range, interval, or tickers.")
        else:
            result_df = pd.DataFrame(results).sort_values("Score")

            price_df = pd.DataFrame(cz_matrix).dropna(axis=0, how="any")
            returns = price_df.pct_change().dropna()
            corr_matrix = returns.corr()

            final_selection = []
            for ticker in result_df["Ticker"]:
                if all(abs(corr_matrix.loc[ticker, sel]) < 0.5 for sel in final_selection):
                    final_selection.append(ticker)
                if len(final_selection) == 5:
                    break

            st.subheader("📌 Top 5 Selected Instruments (Least Correlated)")
            st.write(result_df[result_df["Ticker"].isin(final_selection)])

            st.subheader("🔗 Correlation Matrix (Selected Instruments)")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix.loc[final_selection, final_selection], annot=True, cmap="coolwarm", center=0)
            st.pyplot(fig)
