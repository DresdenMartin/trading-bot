"""Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands."""
import pandas as pd


def add_sma(df: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """Simple moving average."""
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].rolling(window=period).mean()


def add_ema(df: pd.DataFrame, period: int, column: str = "Close") -> pd.Series:
    """Exponential moving average."""
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].ewm(span=period, adjust=False).mean()


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """Relative strength index."""
    if df is None or df.empty or column not in df.columns or len(df) < period + 1:
        return pd.Series(dtype=float)
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """MACD, signal line, and histogram."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame(columns=["MACD", "MACD_Signal", "MACD_Hist"])
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "MACD_Signal": signal_line, "MACD_Hist": hist})


def add_bbands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    column: str = "Close",
) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower."""
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame(columns=["BB_Upper", "BB_Middle", "BB_Lower"])
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return pd.DataFrame({"BB_Upper": upper, "BB_Middle": middle, "BB_Lower": lower})
