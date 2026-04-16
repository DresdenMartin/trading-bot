import os
import sys
import argparse
import requests
from dotenv import load_dotenv


def env_bool(name, default=True):
    v = os.getenv(name, str(default))
    if v is None:
        v = str(default)
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def safe_int(v, default=1):
    try:
        return int(v)
    except Exception:
        return default


def main():
    # load keys/settings
    load_dotenv()
    # parse CLI flags
    parser = argparse.ArgumentParser(description="Place a simple Alpaca order")
    parser.add_argument("--place", action="store_true", help="Place the order instead of dry-run")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt when placing")
    args = parser.parse_args()
    key = os.getenv("ALPACA_KEY")
    secret = os.getenv("ALPACA_SECRET")
    if not key or not secret:
        print("ERROR: ALPACA_KEY and ALPACA_SECRET must be set in the environment or .env file", file=sys.stderr)
        sys.exit(2)

    base = "https://paper-api.alpaca.markets" if env_bool("ALPACA_PAPER") else "https://api.alpaca.markets"

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }

    # check account (can be skipped for local testing)
    if not env_bool("SKIP_ACCOUNT_CHECK", default=False):
        try:
            r = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
            r.raise_for_status()
            acct = r.json()
        except requests.RequestException as e:
            print(f"ERROR: failed to fetch account: {e}", file=sys.stderr)
            sys.exit(3)
        except ValueError:
            print("ERROR: account response not valid JSON", file=sys.stderr)
            sys.exit(3)

        print(f"[ACCOUNT] status={acct.get('status')} buying_power={acct.get('buying_power')}")
    else:
        print("Skipping account check due to SKIP_ACCOUNT_CHECK=true")

    # trade settings
    symbol = os.getenv("TEST_SYMBOL", "AAPL")
    qty = safe_int(os.getenv("TEST_QTY", "1"), 1)
    side = os.getenv("TEST_SIDE", "buy").lower()
    # Determine whether to actually place the order:
    # - explicit CLI `--place` OR env PLACE_ORDER=true will place the order
    # - otherwise dry-run
    place_order_env = env_bool("PLACE_ORDER", default=False)
    dry_run = not (args.place or place_order_env)

    if side not in ("buy", "sell"):
        print("ERROR: TEST_SIDE must be 'buy' or 'sell'", file=sys.stderr)
        sys.exit(4)

    # place order payload
    order = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }

    print(f"[ORDER] prepared: {order} (dry_run={dry_run})")

    if dry_run:
        print("Dry run enabled; not sending order. To place an order set PLACE_ORDER=true or run with --place --yes.")
        return

    # Ask for confirmation unless user passed --yes
    if not args.yes:
        try:
            ans = input("Send order now? Type 'yes' to confirm: ").strip().lower()
        except EOFError:
            ans = ""
        if ans != "yes":
            print("Aborted by user; order not sent.")
            return

    try:
        r = requests.post(f"{base}/v2/orders", headers=headers, json=order, timeout=10)
        r.raise_for_status()
        resp = r.json()
    except requests.RequestException as e:
        print(f"ERROR: failed to place order: {e}", file=sys.stderr)
        sys.exit(5)
    except ValueError:
        print("ERROR: order response not valid JSON", file=sys.stderr)
        sys.exit(5)

    print(f"[ORDER] {resp}")


if __name__ == "__main__":
    main()
