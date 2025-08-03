import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import mplfinance as mpf

from utils.data_loader import fetch_data
from utils.arima_model import forecast_arima
from models.sma import plot_sma_with_mplfinance
from models.cz import chop_zone
from models.ichimoku import ichimoku_cloud
from models.bollinger import plot_bollinger_bands  # <- new
from utils.scraper import get_yahoo_finance_headlines
from utils.llm_sentiment import analyze_sentiment_with_deepseek

# ------------------- App Configuration -------------------
st.set_page_config(page_title="Market Analyzer", layout="wide")
st.title("📈 Multi-Tab Market Analyzer")

# ------------------- Tabs -------------------
tabs = st.tabs(["Instrument Selector", "Instrument Analysis", "Sentiment Analyzer"])

# ------------------- Instrument Selector Tab -------------------
with tabs[0]:
    st.header("🔎 Instrument Selector")

    tickers_input = st.text_area(
        "Enter up to 30 Yahoo Finance tickers (comma-separated)",
        "AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL, JPM, UNH, JNJ, V, MA, PG, XOM, PFE, COST, ORCL, NFLX, META, DIS, KO"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    interval = st.selectbox("Select Time Interval", ["1h", "1d", "1wk", "1mo"], key="interval_selector")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 1, 1), key="selector_start")
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 7, 3), key="selector_end")

    run_button = st.button("Run Evaluation")

    if run_button:
        if not tickers:
            st.error("❌ Please enter at least one valid ticker.")
        else:
            results = []
            cz_matrix = {}

            with st.spinner("📊 Downloading data and evaluating..."):
                for ticker in tickers:
                    df = fetch_data(ticker, str(start_date), str(end_date), interval)

                    if df.empty or len(df) < 50:
                        st.warning(f"⚠️ Not enough data for {ticker}, skipping.")
                        continue

                    cz = chop_zone(df).dropna()
                    if cz.empty:
                        st.warning(f"⚠️ Chop Zone output empty for {ticker}, skipping.")
                        continue

                    last_score = cz.iloc[-1]
                    results.append({"Ticker": ticker, "Score": last_score})
                    series = df["Close"].loc[cz.index]
                    if isinstance(series, pd.Series) and not series.empty:
                        cz_matrix[ticker] = series

            if not results or len(cz_matrix) == 0:
                st.error("❌ No valid instruments processed. Check the date range, interval, or tickers.")
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
                    corr_matrix.loc[final_selection, final_selection],
                    annot=True, cmap="coolwarm", center=0
                )
                st.pyplot(fig)

# ------------------- Instrument Analysis Tab -------------------
with tabs[1]:
    st.header("📈 Instrument Analysis")

    ticker = st.text_input("Instrument (e.g., AAPL, EURUSD=X)", value="NFLX")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 1, 1), key="analysis_start")
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 7, 3), key="analysis_end")

    interval = st.selectbox("Data Interval", ["1h", "1d", "1wk", "1mo"], key="analysis_interval")

    sma_fast = st.number_input("Fast SMA", min_value=1, max_value=50, value=9)
    sma_slow = st.number_input("Slow SMA", min_value=5, max_value=200, value=21)
    forecast_steps = st.number_input(f"Forecast steps ({interval})", min_value=1, max_value=30, value=5)

    show_cz = st.checkbox("Show Chop Zone", value=False)
    show_ichimoku = st.checkbox("Show Ichimoku Cloud", value=False)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=False)

    df = fetch_data(ticker, str(start_date), str(end_date), interval)

    if df.empty:
        st.warning("No data available. Check your ticker or date range.")
    else:
        st.subheader("📉 Price with Simple Moving Averages")
        plot_sma_with_mplfinance(df, lengths=[sma_fast, sma_slow])

        if show_cz:
            cz_series = chop_zone(df).dropna()
            colores = {
                0: "#26C6DA", 1: "#43A047", 2: "#A5D6A7", 3: "#009688",
                4: "#D50000", 5: "#E91E63", 6: "#FF6D00", 7: "#FFB74D", 8: "#FDD835"
            }

            st.subheader("🧩 Chop Zone Indicator")
            cz_fig = go.Figure()
            for i, value in enumerate(cz_series):
                cz_fig.add_trace(go.Bar(
                    x=[cz_series.index[i]],
                    y=[1],
                    marker_color=colores[int(value)],
                    showlegend=False
                ))

            cz_fig.update_layout(
                height=200,
                yaxis=dict(visible=False),
                title="Chop Zone Indicator",
                template="plotly_white"
            )
            st.plotly_chart(cz_fig, use_container_width=True)

        st.subheader("📉 ARIMA Forecast")
        try:
            forecast = forecast_arima(df, steps=forecast_steps, interval=interval)
            valid_df = df.dropna(subset=["Close"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df["Close"], name="Historical Close"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", line=dict(dash="dot", color="red")))

            fig.update_layout(
                title="ARIMA Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Forecast error: {e}")

        if show_ichimoku:
            ichimoku = ichimoku_cloud(df)
            if ichimoku.empty:
                st.warning("Not enough data to compute Ichimoku Cloud.")
            else:
                st.subheader("☁️ Ichimoku Cloud")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="black", width=1), name="Close Price"))
                fig.add_trace(go.Scatter(x=ichimoku.index, y=ichimoku["tenkan_sen"], line=dict(color="purple", dash="dash"), name="Tenkan Sen"))
                fig.add_trace(go.Scatter(x=ichimoku.index, y=ichimoku["kijun_sen"], line=dict(color="orange", dash="dash"), name="Kijun Sen"))
                fig.add_trace(go.Scatter(x=ichimoku.index, y=ichimoku["senkou_span_a"], line=dict(color="green"), name="Senkou Span A"))
                fig.add_trace(go.Scatter(x=ichimoku.index, y=ichimoku["senkou_span_b"], line=dict(color="red"), name="Senkou Span B"))

                fig.add_trace(go.Scatter(x=ichimoku.index, y=ichimoku["senkou_span_a"], fill=None, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=ichimoku.index, y=ichimoku["senkou_span_b"], fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(144,238,144,0.3)', showlegend=True, name="Cloud Area"))

                fig.update_layout(
                    height=500,
                    title="Ichimoku Cloud",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        if show_bollinger:
            st.subheader("📊 Bollinger Bands")
            try:
                fig = plot_bollinger_bands(df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Bollinger error: {e}")

# ------------------- Placeholder Sentiment Tab -------------------
with tabs[2]:
    st.header("🧠 Sentiment Analyzer")

    company_name = st.text_input("Enter a Company Name (e.g., Apple, Amazon, Tesla)", value="Apple")

    analyze_btn = st.button("Run Sentiment Analysis")

    if analyze_btn:
        if not company_name.strip():
            st.error("⚠️ Please enter a valid company name.")
        else:
            from utils.llm_sentiment import analyze_sentiment_with_deepseek

            with st.spinner("🔍 Fetching news and analyzing sentiment..."):
                sentiment_result = analyze_sentiment_with_deepseek(company_name.strip())

            # Display result
            if sentiment_result.startswith("❗") or sentiment_result.startswith("❌"):
                st.warning(sentiment_result)
            else:
                st.success("✅ Analysis complete!")
                st.markdown(sentiment_result)

