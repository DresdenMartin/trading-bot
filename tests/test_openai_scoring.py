import os
import json
from unittest.mock import patch

os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'

import eod_fetcher


@patch('eod_fetcher.call_chat_completion')
def test_aggregate_news_attaches_gpt_scores(call_mock):
    # prepare fake articles with recent timestamps so they are within the 48h window
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    a1_ts = (now - timedelta(hours=2)).isoformat()
    a2_ts = (now - timedelta(hours=1)).isoformat()
    articles = [
        {'source': 'polygon', 'title': 'Acme posts record quarter', 'description': 'Revenue up', 'published_at': a1_ts},
        {'source': 'polygon', 'title': 'Acme issues warning', 'description': 'Guidance lowered', 'published_at': a2_ts},
    ]
    # first article positive, second negative
    call_mock.side_effect = [
        '{"sentiment": 0.7, "confidence": 0.8, "catalysts": ["earnings"]}',
        '{"sentiment": -0.6, "confidence": 0.6, "catalysts": ["guidance"]}',
    ]

    # patch fetchers in eod_fetcher so the aggregator uses our test articles
    os.environ['OPENAI_API_KEY'] = 'fake'
    res_articles, agg = eod_fetcher.aggregate_news_for_symbol('ACME', limit=10, hours=48, seed_articles=articles)
    # ensure returned articles have gpt_score_100 and gpt_sent
    assert len(res_articles) == 2
    for a in res_articles:
        assert 'gpt_score_100' in a
        assert isinstance(a['gpt_score_100'], int)
        assert 0 <= a['gpt_score_100'] <= 100
        assert 'gpt_sent' in a
        assert -1.0 <= a['gpt_sent'] <= 1.0
    # Agg should be numeric and reflect average combined; not null
    assert isinstance(agg, float)


@patch('eod_fetcher.call_chat_completion')
def test_openai_fallback_to_cached_batch(call_mock):
    # Simulate exception by raising during call
    call_mock.side_effect = Exception('client init failed')
    os.environ['OPENAI_API_KEY'] = 'fake'
    # ensure aggregator still runs without raising by patching fetchers to return at least one recent article
    from datetime import datetime, timezone
    recent = datetime.now(timezone.utc).isoformat()
    fake = [{'source': 'polygon', 'title': 'noop', 'description': 'noop', 'published_at': recent}]
    res_articles, agg = eod_fetcher.aggregate_news_for_symbol('ACME', limit=5, hours=48, seed_articles=fake)
    assert isinstance(res_articles, list)
    assert isinstance(agg, float)
