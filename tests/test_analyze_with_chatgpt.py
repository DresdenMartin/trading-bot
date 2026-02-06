import os
from unittest.mock import patch
import json
import analyze_with_chatgpt

os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'


@patch('analyze_with_chatgpt.call_chat_completion')
def test_analyze_summary_with_mock(call_mock):
    call_mock.return_value = '{"summary":"ok","signals":{},"suggested_action":"hold","confidence":0.5,"recommended_size":"1%","rationale":"test"}'

    os.environ['OPENAI_API_KEY'] = 'fake'
    res = analyze_with_chatgpt.analyze_summary('ACME', {'foo': 'bar'})
    assert isinstance(res, dict)
    assert res.get('summary') == 'ok'
    assert 'suggested_action' in res


@patch('analyze_with_chatgpt.call_chat_completion')
def test_analyze_portfolio_model_path(call_mock):
    # Mock two-symbol response
    arr = [
        {'symbol': 'A', 'score': 80, 'rationale': 'good'},
        {'symbol': 'B', 'score': 20, 'rationale': 'bad'},
    ]
    call_mock.return_value = json.dumps(arr)

    os.environ['OPENAI_API_KEY'] = 'fake'
    items = [
        {'symbol': 'A', 'metrics': {'news_aggregate_sentiment': 0.5}, 'news_articles': [], 'analyst_entries': []},
        {'symbol': 'B', 'metrics': {'news_aggregate_sentiment': -0.5}, 'news_articles': [], 'analyst_entries': []},
    ]
    out = analyze_with_chatgpt.analyze_portfolio_short_term(items)
    assert isinstance(out, list)
    assert out[0]['symbol'] == 'A'
    assert out[1]['symbol'] == 'B'


def test_analyze_portfolio_heuristic_fallback():
    # ensure heuristic runs when OPENAI_API_KEY not set
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    items = [
        {'symbol': 'A', 'metrics': {'news_aggregate_sentiment': 0.0, 'rsi_14': 25}, 'news_articles': ['n'], 'analyst_entries': {}},
        {'symbol': 'B', 'metrics': {'news_aggregate_sentiment': 0.0, 'rsi_14': 75}, 'news_articles': [], 'analyst_entries': {}},
    ]
    out = analyze_with_chatgpt.analyze_portfolio_short_term(items)
    assert isinstance(out, list)
    assert 'score' in out[0]
    assert 'rationale' in out[0]
