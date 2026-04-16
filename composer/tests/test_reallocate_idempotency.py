import json
from types import SimpleNamespace
from scheduled_trader import analyze_mag7_and_invest, analyze_top10_and_invest


def test_reallocate_idempotency(monkeypatch):
    # Fake account and positions to force a buy for 'BAR'
    fake_account = {'cash': '10000', 'buying_power': '10000'}
    fake_positions = []

    def fake_discover(*args, **kwargs):
        return ['BAR']

    monkeypatch.setattr('scheduled_trader.discover_top_symbols_via_news', fake_discover)

    def fake_fetch_eod(symbols, output_dir='./data'):
        import pandas as pd
        df = pd.DataFrame({'Close': [50.0, 50.0]})
        return {'BAR': df}

    monkeypatch.setattr('scheduled_trader.fetch_eod', fake_fetch_eod)

    def fake_analyze(items):
        return [{'symbol': 'BAR', 'score': 90}]

    monkeypatch.setattr('analyze_with_chatgpt.analyze_portfolio_short_term', fake_analyze, raising=False)

    monkeypatch.setattr('scheduled_trader.fetch_alpaca_account_and_positions', lambda: (fake_account, fake_positions))

    # Executor that tracks client_order_id and returns success first time, 'already_exists' on repeat
    seen_coids = set()
    calls = []

    def order_executor(url, headers=None, json=None, timeout=10):
        calls.append(json)
        coid = (json or {}).get('client_order_id')
        if coid in seen_coids:
            return {'error': 'already_exists', 'client_order_id': coid}
        seen_coids.add(coid)
        return {'id': 'ok', 'client_order_id': coid, 'status': 'filled'}

    # enable PLACE_ORDER
    monkeypatch.setenv('PLACE_ORDER', '1')
    # set dummy Alpaca keys so the code proceeds to placement (tests inject executor)
    monkeypatch.setenv('ALPACA_KEY', 'testkey')
    monkeypatch.setenv('ALPACA_SECRET', 'testsecret')

    # mock scheduled_trader.requests.get used to fetch account/positions
    def fake_get(url, headers=None, timeout=10):
        class R:
            def raise_for_status(self):
                return None

            def json(self_inner):
                if url.endswith('/v2/account'):
                    return {'cash': '10000', 'buying_power': '10000'}
                if url.endswith('/v2/positions'):
                    return []
                return {}

        return R()

    monkeypatch.setattr('scheduled_trader.requests.get', fake_get)

    # First run: executor should be called and coid recorded
    assert analyze_top10_and_invest is analyze_mag7_and_invest

    res1 = analyze_mag7_and_invest(reallocate_full_arg=True, place_order_arg=True, invest_yes_arg=True, order_executor=order_executor)
    assert any(p['action'] == 'buy' for p in res1.get('placed', []))
    assert len(calls) >= 1
    first_coid = None
    for c in calls:
        if 'client_order_id' in (c or {}):
            first_coid = c.get('client_order_id')
            break
    assert first_coid is not None

    # Clear placed/calls tracking for second run but keep the seen_coids to simulate server-side persistence
    calls.clear()

    # Second run: should attempt same client_order_id, executor returns 'already_exists' for that coid
    res2 = analyze_mag7_and_invest(reallocate_full_arg=True, place_order_arg=True, invest_yes_arg=True, order_executor=order_executor)
    # ensure a buy entry exists, and its resp indicates idempotent handling
    buys = [p for p in res2.get('placed', []) if p.get('action') == 'buy']
    assert buys
    # check executor was called and that it received the same client_order_id
    coids_second_run = [c.get('client_order_id') for c in calls if c and 'client_order_id' in c]
    assert first_coid in coids_second_run
    # check that response shows already_exists error for repeated coid
    assert any((p.get('resp') or {}).get('error') == 'already_exists' or (p.get('resp') or {}).get('client_order_id') == first_coid for p in res2.get('placed', []))
