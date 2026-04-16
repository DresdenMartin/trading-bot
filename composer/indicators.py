import pandas as pd


def add_sma(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return df['Close'].rolling(window=window).mean()


def add_ema(df: pd.DataFrame, span: int = 14) -> pd.Series:
    return df['Close'].ewm(span=span, adjust=False).mean()


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_macd(df: pd.DataFrame, span_short: int = 12, span_long: int = 26, span_signal: int = 9) -> pd.DataFrame:
    ema_short = df['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({'MACD': macd, 'MACD_Signal': signal, 'MACD_Hist': hist})


def add_bbands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    sma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({'BB_upper': upper, 'BB_lower': lower, 'BB_mid': sma})
