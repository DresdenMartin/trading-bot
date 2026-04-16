Safe order executor

This small helper lets you rehearse or run order placement in a controlled way.

Usage example

```python
from tools.safe_order_executor import SafeOrderExecutor
from scheduled_trader import analyze_mag7_and_invest

# Create an executor that logs but does not forward
exe = SafeOrderExecutor(enabled=False, paper=True)  # logs to project logs directory by default
# Run a controlled reallocation (dry run will still call executor, which will only log)
res = analyze_mag7_and_invest(reallocate_full_arg=True, place_order_arg=True, invest_yes_arg=True, order_executor=exe.exec)
print(res)
```

To actually forward requests to Alpaca, set `enabled=True` and ensure `ALPACA_KEY` and `ALPACA_SECRET` environment variables are set. The `paper` flag controls whether the paper API endpoint is used.

Notes
- The executor will always write payloads to `log_path` for review.
- When `enabled=False` the executor returns a simulated success dict so calling code proceeds as if an order succeeded.
- This is intended for rehearsal and light manual runs only. Always double-check payloads and audit logs before enabling live forwarding.
