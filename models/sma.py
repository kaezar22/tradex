import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import streamlit as st


def Media_movil_simple(data: pd.DataFrame, longitud: int, columna="Close") -> pd.Series:
    """
    Calculate simple moving average.
    """
    return data[columna].rolling(window=longitud).mean().reindex(data.index)


def plot_sma_with_mplfinance(df: pd.DataFrame, lengths=[9, 21], forecast: pd.Series = None) -> None:
    """
    Plot candlestick chart with SMAs and optional ARIMA forecast using mplfinance.
    Streamlit-compatible using st.pyplot().
    """

    # 🧼 Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 🧼 Handle repeated ticker column names
    if df.columns.tolist() == ['NFLX'] * 5:
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 🧼 Clean numeric data
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df.index)

    if df.empty:
        st.warning("📉 Not enough clean data to plot SMA.")
        return

    # ✅ Calculate SMAs
    plots = []
    for length in lengths:
        sma = Media_movil_simple(df, longitud=length)
        plots.append(
            mpf.make_addplot(
                sma,
                label=f"SMA {length}",
                color="green" if length < 20 else "blue",
                linestyle="solid"
            )
        )

    # ✅ Add ARIMA forecast if available
    if forecast is not None and not forecast.empty:
        # Combine historical and forecast indices
        full_index = df.index.union(forecast.index)

        # Extend df with full index and forward-fill values
        df_extended = df.reindex(full_index)
        df_extended.fillna(method="ffill", inplace=True)

        # Align forecast with the extended index
        forecast_aligned = forecast.reindex(df_extended.index)

        # Add forecast line to plots
        forecast_plot = mpf.make_addplot(
            forecast_aligned,
            color="red",
            linestyle="dashdot",
            width=2,
            label="ARIMA Forecast"
        )
        plots.append(forecast_plot)

    else:
        df_extended = df
    # 🐞 DEBUGGING — add just before plotting
    # st.write("📉 DEBUG - df_extended shape:", df_extended.shape)
    if forecast is not None and not forecast.empty:
        st.write("Forecast aligned shape:", forecast_aligned.shape)
        st.write("Forecast aligned index:", forecast_aligned.index)
        st.write("df_extended index:", df_extended.index)

    # ✅ Plot everything using mplfinance
    fig, _ = mpf.plot(
        df_extended,
        type='candle',
        style='yahoo',
        volume=True,
        addplot=plots,
        figsize=(14, 8),
        returnfig=True,
        title="Price Chart with SMAs and Forecast"
    )

    st.pyplot(fig)
