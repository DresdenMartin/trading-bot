from run_eod import compute_candidate_score


def make_analysis(suggested='hold', confidence=0.6):
    return {'suggested_action': suggested, 'confidence': confidence}


def test_score_prefers_buy_over_hold():
    a_buy = make_analysis('buy', 0.6)
    a_hold = make_analysis('hold', 0.9)
    metrics = {'close': 100.0, 'ema_14': 95.0, 'sma_14': 94.0, 'rsi_14': 50.0, 'news_aggregate_sentiment': 0.1}
    sb = compute_candidate_score(a_buy, metrics)
    sh = compute_candidate_score(a_hold, metrics)
    assert sb > sh


def test_score_reflects_confidence():
    a_low = make_analysis('hold', 0.2)
    a_high = make_analysis('hold', 0.8)
    metrics = {'close': 100.0, 'ema_14': 95.0, 'sma_14': 94.0, 'rsi_14': 50.0, 'news_aggregate_sentiment': 0.0}
    assert compute_candidate_score(a_high, metrics) > compute_candidate_score(a_low, metrics)
