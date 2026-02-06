import pandas as pd


def simple_signals(df: pd.DataFrame) -> dict:
    """Return simple signals based on SMA/EMA crossover and RSI thresholds."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    signals = {}
    # SMA/EMA crossover
    if latest.get('SMA_14') is not None and latest.get('EMA_14') is not None:
        if prev['SMA_14'] < prev['EMA_14'] and latest['SMA_14'] > latest['EMA_14']:
            signals['sma_ema'] = 'bullish_crossover'
        elif prev['SMA_14'] > prev['EMA_14'] and latest['SMA_14'] < latest['EMA_14']:
            signals['sma_ema'] = 'bearish_crossover'
    # RSI thresholds
    rsi = latest.get('RSI_14')
    if rsi is not None:
        if rsi < 30:
            signals['rsi'] = 'oversold'
        elif rsi > 70:
            signals['rsi'] = 'overbought'
    return signals
