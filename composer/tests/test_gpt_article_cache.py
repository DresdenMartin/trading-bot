import os
from unittest.mock import patch, MagicMock
os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'

import eod_fetcher


def fake_html_response(html):
    m = MagicMock()
    m.status_code = 200
    m.text = html
    m.raise_for_status = lambda: None
    return m


@patch('eod_fetcher.requests.get')
@patch('eod_fetcher.call_chat_completion')
def test_article_fetch_and_cache(call_mock, requests_get):
    # simple HTML with article tag
    html = '<html><body><article><p>Paragraph one.</p><p>Paragraph two.</p></article></body></html>'
    requests_get.return_value = fake_html_response(html)

    # mock OpenAI to return a score
    call_mock.return_value = '{"sentiment": 0.54, "confidence": 0.8, "catalysts": ["earnings beat"]}'

    os.environ['OPENAI_API_KEY'] = 'fake'
    # remove any existing cache entry file for a clean test
    try:
        os.remove('.cache/gpt_article_cache.json')
    except Exception:
        pass
    # clear in-memory cache in module in case prior tests populated it
    try:
        eod_fetcher._GPT_ARTICLE_CACHE.clear()
    except Exception:
        pass

    url = 'https://example.com/article/1'
    articles = [{'source': 'polygon', 'title': 't1', 'url': url, 'published_at': '2025-09-16T10:00:00Z'}]

    res, agg = eod_fetcher.aggregate_news_for_symbol('ACME', limit=5, hours=48, seed_articles=articles)
    assert len(res) == 1
    a = res[0]
    assert 'article_text' in a and 'Paragraph one' in a['article_text']
    assert a['gpt_score_100'] == 77
    assert abs(a['gpt_sentiment'] - 0.54) < 1e-6
    assert a['gpt_confidence'] == 0.8
    assert a['gpt_catalysts'] == ['earnings beat']

    # Now simulate second run: OpenAI should not be called again because of cache
    call_mock.reset_mock()
    res2, agg2 = eod_fetcher.aggregate_news_for_symbol('ACME', limit=5, hours=48, seed_articles=articles)
    assert len(res2) == 1
    a2 = res2[0]
    assert a2['gpt_score_100'] == 77
    assert abs(a2['gpt_sentiment'] - 0.54) < 1e-6
    # OpenAI constructor should not be invoked because cache used
    call_mock.assert_not_called()
