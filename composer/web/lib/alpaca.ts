const ALPACA_BASE =
  process.env.ALPACA_PAPER === "false"
    ? "https://api.alpaca.markets"
    : "https://paper-api.alpaca.markets";

const ALPACA_DATA_BASE = "https://data.alpaca.markets";

export { ALPACA_BASE, ALPACA_DATA_BASE };

// ── Types ────────────────────────────────────────────────────────────────────

export interface AlpacaAccount {
  id: string;
  status: string;
  equity: string;
  buying_power: string;
  cash: string;
  portfolio_value: string;
  daytrade_count: number;
  last_equity: string;
  pattern_day_trader: boolean;
}

export interface AlpacaPosition {
  symbol: string;
  qty: string;
  avg_entry_price: string;
  market_value: string;
  cost_basis: string;
  unrealized_pl: string;
  unrealized_plpc: string;
  current_price: string;
  lastday_price: string;
  change_today: string;
  side: "long" | "short";
}

export interface AlpacaOrder {
  id: string;
  client_order_id: string;
  symbol: string;
  qty: string;
  filled_qty: string;
  side: "buy" | "sell";
  type: "market" | "limit" | "stop" | "stop_limit";
  time_in_force: "day" | "gtc" | "opg" | "cls" | "ioc" | "fok";
  status: string;
  created_at: string;
  filled_at: string | null;
  filled_avg_price: string | null;
  limit_price: string | null;
  extended_hours: boolean;
}

export interface AlpacaBar {
  t: string; // RFC-3339 timestamp
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
  vw: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function alpacaHeaders() {
  return {
    "APCA-API-KEY-ID": process.env.ALPACA_KEY ?? "",
    "APCA-API-SECRET-KEY": process.env.ALPACA_SECRET ?? "",
    "Content-Type": "application/json",
  };
}

// ── API calls ────────────────────────────────────────────────────────────────

export async function fetchAccount(): Promise<AlpacaAccount | null> {
  try {
    const res = await fetch(`${ALPACA_BASE}/v2/account`, {
      headers: alpacaHeaders(),
      next: { revalidate: 30 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function fetchPositions(): Promise<AlpacaPosition[]> {
  try {
    const res = await fetch(`${ALPACA_BASE}/v2/positions`, {
      headers: alpacaHeaders(),
      next: { revalidate: 30 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchOrders(
  status: "open" | "closed" | "all" = "all",
  limit = 50
): Promise<AlpacaOrder[]> {
  try {
    const url = new URL(`${ALPACA_BASE}/v2/orders`);
    url.searchParams.set("status", status);
    url.searchParams.set("limit", String(limit));
    const res = await fetch(url.toString(), {
      headers: alpacaHeaders(),
      next: { revalidate: 10 },
    });
    if (!res.ok) return [];
    return res.json();
  } catch {
    return [];
  }
}

export async function fetchLatestBars(
  symbols: string[]
): Promise<Record<string, AlpacaBar>> {
  if (!symbols.length) return {};
  try {
    const url = new URL(`${ALPACA_DATA_BASE}/v2/stocks/bars/latest`);
    url.searchParams.set("symbols", symbols.join(","));
    url.searchParams.set("feed", "iex");
    const res = await fetch(url.toString(), {
      headers: alpacaHeaders(),
      next: { revalidate: 15 },
    });
    if (!res.ok) return {};
    const data = await res.json();
    return data.bars ?? {};
  } catch {
    return {};
  }
}
