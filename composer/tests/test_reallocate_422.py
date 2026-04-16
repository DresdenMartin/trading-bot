import json
from types import SimpleNamespace
from scheduled_trader import analyze_mag7_and_invest, analyze_top10_and_invest


def test_reallocate_uses_notional_on_422(tmp_path, monkeypatch):
    # Prepare a fake account with cash and a single position that will trigger a buy
    fake_account = {'cash': '1000', 'buying_power': '1000'}
    # fake position: holds None so target allocation requires buying
    fake_positions = [{'symbol': 'DUMMY', 'qty': '0'}]

    # Create data such that the ranked list includes 'FOO' and we'll need to buy it
    def fake_discover(*args, **kwargs):
        return ['FOO']

    monkeypatch.setattr('scheduled_trader.discover_top_symbols_via_news', fake_discover)

    # Provide fetch_eod returning price for FOO
    def fake_fetch_eod(symbols, output_dir='./data'):
        import pandas as pd
        df = pd.DataFrame({'Close': [10.0, 10.0]})
        return {'FOO': df}

    monkeypatch.setattr('scheduled_trader.fetch_eod', fake_fetch_eod)

    # ensure PLACE_ORDER is enabled for this test
    monkeypatch.setenv('PLACE_ORDER', '1')

    # Mock analyze_portfolio_short_term to return a single top-ranked entry
    def fake_analyze(items):
        return [{'symbol': 'FOO', 'score': 90}]

    monkeypatch.setattr('analyze_with_chatgpt.analyze_portfolio_short_term', fake_analyze, raising=False)

    # Mock fetch_alpaca_account_and_positions to return our fake acct/positions
    monkeypatch.setattr('scheduled_trader.fetch_alpaca_account_and_positions', lambda: (fake_account, fake_positions))

    # Track calls to order executor
    calls = []

    class RespErr(Exception):
        def __init__(self, response=None):
            self.response = response

    class FakeResp:
        def __init__(self, status_code=200, json_body=None, text=''):
            self.status_code = status_code
            self._json = json_body or {}
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                err = RespErr()
                err.response = SimpleNamespace(status_code=self.status_code, text=self.text)
                raise err

        def json(self):
            return self._json

    # executor: first call (qty payload) will raise 422, second call (notional) returns success
    def order_executor(url, headers=None, json=None, timeout=10):
        calls.append({'url': url, 'json': json})
        # if payload contains 'qty', simulate 422
        if 'qty' in (json or {}):
            return FakeResp(status_code=422, json_body={'error': 'unprocessable', 'body': '422 response'}, text='unprocessable')
        # notional attempt returns success dict
        return {'id': 'ok', 'status': 'filled'}

    # Ensure PLACE_ORDER is enabled via arg
    # ensure legacy alias still points to the new implementation
    assert analyze_top10_and_invest is analyze_mag7_and_invest

    res = analyze_mag7_and_invest(reallocate_full_arg=True, place_order_arg=True, invest_yes_arg=True, order_executor=order_executor)

    # Validate that two calls were made for the buy (qty then notional)
    # locate buy call payloads
    buy_calls = [c for c in calls if '/v2/orders' in c['url']]
    assert len(buy_calls) >= 2
    # first should contain 'qty', second should contain 'notional'
    assert 'qty' in buy_calls[0]['json']
    assert 'notional' in buy_calls[1]['json']

    # verify placed entries indicate a buy and contain resp
    placed = res.get('placed') or []
    assert any(p['action'] == 'buy' for p in placed)
    # ensure audit file was attempted (res contains placed responses)
    assert any(p['resp'] is not None for p in placed)
