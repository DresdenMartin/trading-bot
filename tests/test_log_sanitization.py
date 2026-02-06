import os
import json
from tools.safe_order_executor import SafeOrderExecutor


def test_safe_order_executor_masks_headers(tmp_path):
    logf = tmp_path / 'safe.log'
    exe = SafeOrderExecutor(enabled=False, log_path=str(logf))
    payload_headers = {
        'APCA-API-KEY-ID': 'PK1234567890SECRET',
        'APCA-API-SECRET-KEY': 'SOMEVERYSECRET0000',
        'Content-Type': 'application/json',
    }
    payload_json = {'symbol': 'AAPL', 'qty': 1, 'client_order_id': '2025-09-17::AAPL', 'api_key_like': 'SHOULD_BE_MASKED'}
    exe.exec('https://paper-api.alpaca.markets/v2/orders', headers=payload_headers, json=payload_json)
    data = logf.read_text()
    assert 'PK1234567890SECRET' not in data
    assert 'SOMEVERYSECRET0000' not in data
    assert 'SHOULD_BE_MASKED' not in data
    # sanitized markers present
    assert '***REDACTED***' in data