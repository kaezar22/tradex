# indicators/cz.py

import pandas as pd
import numpy as np


def chop_zone(df: pd.DataFrame, longitud: int = 30, longitud_ema: int = 34, columna: str = "Close") -> pd.Series:
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
    angulo_ema0 = np.rad2deg(np.arccos(1 / c_ema)).round()
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

    result = pd.Series([np.nan] * max(longitud, longitud_ema) +
                       CZ.tolist(), index=df.index, name="CZ")
    return result
