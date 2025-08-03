import pandas as pd


def ichimoku_cloud(df: pd.DataFrame, periodo_tenkan: int = 9, periodo_kijun: int = 26, offset: bool = False) -> pd.DataFrame:
    High, Low = df["High"], df["Low"]

    tenkan_sen = (High.rolling(window=periodo_tenkan).max() +
                  Low.rolling(window=periodo_tenkan).min()) / 2
    kijun_sen = (High.rolling(window=periodo_kijun).max() +
                 Low.rolling(window=periodo_kijun).min()) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (High.rolling(window=periodo_kijun * 2).max() +
                     Low.rolling(window=periodo_kijun * 2).min()) / 2
    chikou_span = df["Close"].shift(-periodo_kijun)

    cloud = pd.DataFrame({
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a.shift(periodo_kijun) if not offset else senkou_span_a,
        "senkou_span_b": senkou_span_b.shift(periodo_kijun) if not offset else senkou_span_b,
        "chikou_span": chikou_span
    }, index=df.index)

    return cloud
