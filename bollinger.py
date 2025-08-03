import pandas as pd
import plotly.graph_objects as go
from copy import deepcopy

def compute_bollinger_bands(df: pd.DataFrame, longitud: int = 20, std_dev: float = 2.0, ddof: int = 0, columna: str = "Close") -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    """
    data = deepcopy(df)
    rolling = data[columna].rolling(window=longitud, min_periods=longitud)
    data["MA"] = rolling.mean()
    calc_intermedio = std_dev * rolling.std(ddof=ddof)
    data["BB_Up"] = data["MA"] + calc_intermedio
    data["BB_Down"] = data["MA"] - calc_intermedio
    return data

def plot_bollinger_bands(df: pd.DataFrame) -> go.Figure:
    """
    Plot Bollinger Bands using Plotly.
    """
    df_bb = compute_bollinger_bands(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["Close"], name="Close", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["MA"], name="MA", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["BB_Up"], name="Upper Band", line=dict(color="green", dash="dot")))
    fig.add_trace(go.Scatter(x=df_bb.index, y=df_bb["BB_Down"], name="Lower Band", line=dict(color="red", dash="dot")))

    fig.update_layout(
        title="Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig
