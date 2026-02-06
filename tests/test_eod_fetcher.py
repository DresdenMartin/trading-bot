import os
import json
import types
import pytest

from unittest.mock import patch, MagicMock

os.environ['OPENAI_MODEL'] = 'gpt-4o-mini'

import eod_fetcher


def test_simple_sentiment_keywords():
    assert eod_fetcher._simple_sentiment('strong beat and outperform') > 0
    assert eod_fetcher._simple_sentiment('huge downgrade and miss') < 0


@patch('eod_fetcher.call_chat_completion')
def test_openai_sentiment_mock(call_mock):
    call_mock.return_value = '{"score": 0.5}'
    os.environ['OPENAI_API_KEY'] = 'fake'
    s = eod_fetcher._openai_sentiment('this is good')
    assert isinstance(s, float)
    assert s == 0.5


@patch('eod_fetcher.requests.get')
def test_fetch_polygon_analyst_ratings_preferred(mock_get):
    # simulate a polygon analyst ratings response
    mock_get.return_value = MagicMock(status_code=200)
    mock_get.return_value.json.return_value = {
        'results': [
            {'rating': 'Buy', 'date': '2025-09-16'},
            {'rating': 'Hold', 'date': '2025-09-15'},
            {'rating': 'Buy', 'date': '2025-09-14'},
        ]
    }
    os.environ['POLYGON_KEY'] = 'fake'
    # ensure cache doesn't leak between tests
    try:
        import os as _os

        _os.remove(os.path.join('.', '.cache', 'polygon_analysts.json'))
    except Exception:
        pass
    out = eod_fetcher.fetch_analyst_sentiment('ACME')
    assert isinstance(out, dict)
    assert 'counts' in out
    assert out['counts']['buy'] == 2
    # ensure entries list exists and contains expected keys
    assert 'entries' in out
    assert isinstance(out['entries'], list)
    assert out['entries']
    e = out['entries'][0]
    assert 'rating_text' in e
    assert 'firm' in e
    assert 'date' in e


@patch('eod_fetcher.requests.get')
def test_polygon_analyst_date_normalization_epoch(mock_get):
    # epoch timestamp
    mock_get.return_value = MagicMock(status_code=200)
    mock_get.return_value.json.return_value = {
        'results': [
            {'rating': 'Buy', 'date': 1694822400},
        ]
    }
    os.environ['POLYGON_KEY'] = 'fake'
    try:
        import os as _os

        _os.remove(os.path.join('.', '.cache', 'polygon_analysts.json'))
    except Exception:
        pass
    out = eod_fetcher.fetch_polygon_analyst_ratings('ACME')
    assert isinstance(out, dict)
    assert 'entries' in out
    assert out['entries'][0]['date'] is not None


@patch('eod_fetcher.requests.get')
def test_polygon_analyst_date_normalization_iso(mock_get):
    mock_get.return_value = MagicMock(status_code=200)
    mock_get.return_value.json.return_value = {
        'results': [
            {'rating': 'Hold', 'date': '2025-09-16T12:34:56Z'},
        ]
    }
    os.environ['POLYGON_KEY'] = 'fake'
    try:
        import os as _os

        _os.remove(os.path.join('.', '.cache', 'polygon_analysts.json'))
    except Exception:
        pass
    out = eod_fetcher.fetch_polygon_analyst_ratings('ACME')
    assert out['entries'][0]['date'] is not None


@patch('eod_fetcher.requests.get')
def test_polygon_analyst_date_normalization_other(mock_get):
    # other common format
    mock_get.return_value = MagicMock(status_code=200)
    mock_get.return_value.json.return_value = {
        'results': [
            {'rating': 'Sell', 'date': 'Sep 16 2025 12:34:56'},
        ]
    }
    os.environ['POLYGON_KEY'] = 'fake'
    try:
        import os as _os

        _os.remove(os.path.join('.', '.cache', 'polygon_analysts.json'))
    except Exception:
        pass
    out = eod_fetcher.fetch_polygon_analyst_ratings('ACME')
    assert out['entries'][0]['date'] is not None
