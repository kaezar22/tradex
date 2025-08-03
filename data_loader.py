import yfinance as yf
import pandas as pd


def fetch_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Fetches and cleans historical market data from Yahoo Finance.

    :param ticker: The instrument symbol, e.g. "EURUSD=X", "AAPL"
    :param start_date: Start date in "YYYY-MM-DD" format
    :param end_date: End date in "YYYY-MM-DD" format
    :param interval: Interval for data, e.g. "1h", "1d", "1wk", "1mo"
    :return: Cleaned OHLCV DataFrame or empty DataFrame if error
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         interval=interval, progress=False)

        if df.empty:
            return df

        # Fix MultiIndex or duplicated column issues
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.columns.tolist() == [ticker] * 5:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Clean numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
