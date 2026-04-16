"use client";

import { useState, useEffect, useCallback } from "react";
import {
  DollarSign,
  TrendingUp,
  Wallet,
  Layers,
  RefreshCw,
  Rocket,
  PauseCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EquityChart, type ChartPoint } from "@/components/charts/equity-chart";
import { cn } from "@/lib/utils";

interface ActiveStrategy {
  id: string;
  strategy_id: string;
  status: "active" | "paused";
  last_run_at: string | null;
  strategies: { name: string } | null;
}

interface AlpacaOrder {
  id: string;
  symbol: string;
  side: string;
  qty: string;
  filled_qty: string;
  filled_avg_price: string | null;
  status: string;
  created_at: string;
}

// ── Types ────────────────────────────────────────────────────────────────────

interface AlpacaAccount {
  equity: string;
  last_equity: string;
  buying_power: string;
  cash: string;
  portfolio_value: string;
  status: string;
  pattern_day_trader: boolean;
}

interface AlpacaPosition {
  symbol: string;
  qty: string;
  side: string;
  market_value: string;
  unrealized_pl: string;
  unrealized_plpc: string;
  current_price: string;
  avg_entry_price: string;
  change_today: string;
}

// ── Formatters ───────────────────────────────────────────────────────────────

function fmtUSD(v: string | number | null | undefined): string {
  if (v == null) return "—";
  const n = typeof v === "string" ? parseFloat(v) : v;
  if (!isFinite(n)) return "—";
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function fmtPct(ratio: string | number | null | undefined): string {
  if (ratio == null) return "—";
  const n = typeof ratio === "string" ? parseFloat(ratio) : ratio;
  if (!isFinite(n)) return "—";
  const pct = n * 100;
  return `${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

function fmtQty(v: string | number | null | undefined): string {
  if (v == null) return "—";
  const n = typeof v === "string" ? parseFloat(v) : v;
  if (!isFinite(n)) return "—";
  return n % 1 === 0 ? n.toFixed(0) : n.toFixed(4);
}

// ── Period selector ───────────────────────────────────────────────────────────

const PERIODS = [
  { label: "1D", value: "1D" },
  { label: "1W", value: "1W" },
  { label: "1M", value: "1M" },
] as const;
type Period = (typeof PERIODS)[number]["value"];

// ── Page ─────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [account, setAccount] = useState<AlpacaAccount | null>(null);
  const [positions, setPositions] = useState<AlpacaPosition[]>([]);
  const [history, setHistory] = useState<ChartPoint[]>([]);
  const [baseValue, setBaseValue] = useState<number | null>(null);
  const [period, setPeriod] = useState<Period>("1M");
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [activeStrategies, setActiveStrategies] = useState<ActiveStrategy[]>([]);
  const [recentOrders, setRecentOrders] = useState<AlpacaOrder[]>([]);

  // Fetch active strategies + recent orders
  const fetchMeta = useCallback(async () => {
    try {
      const [activeRes, ordersRes] = await Promise.all([
        fetch("/api/active-strategies").catch(() => null),
        fetch("/api/alpaca/orders").catch(() => null),
      ]);
      if (activeRes?.ok) setActiveStrategies(await activeRes.json());
      if (ordersRes?.ok) setRecentOrders(await ordersRes.json());
    } catch {}
  }, []);

  // Fetch account + positions (fast, no chart reload)
  const fetchCore = useCallback(async () => {
    const [acctRes, posRes] = await Promise.all([
      fetch("/api/alpaca/account"),
      fetch("/api/alpaca/positions"),
    ]);
    if (!acctRes.ok) {
      const body = await acctRes.json().catch(() => ({}));
      throw new Error(body.error ?? `Account fetch failed (${acctRes.status})`);
    }
    const [acct, pos] = await Promise.all([
      acctRes.json(),
      posRes.ok ? posRes.json() : [],
    ]);
    setAccount(acct);
    setPositions(Array.isArray(pos) ? pos : []);
  }, []);

  // Fetch chart history (separate so period changes don't reset positions)
  const fetchHistory = useCallback(async (p: Period) => {
    setChartLoading(true);
    try {
      const res = await fetch(`/api/alpaca/history?period=${p}`);
      if (res.ok) {
        const data = await res.json();
        setHistory(data.points ?? []);
        setBaseValue(data.base_value ?? null);
      }
    } finally {
      setChartLoading(false);
    }
  }, []);

  // Full refresh
  const refresh = useCallback(
    async (p: Period = period) => {
      setLoading(true);
      setError(null);
      try {
        await Promise.all([fetchCore(), fetchHistory(p), fetchMeta()]);
        setLastUpdated(
          new Date().toLocaleTimeString("en-US", {
            hour: "numeric",
            minute: "2-digit",
          })
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [period, fetchCore, fetchHistory, fetchMeta]
  );

  // Initial load
  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Period change — only reload chart
  const handlePeriod = (p: Period) => {
    setPeriod(p);
    fetchHistory(p);
  };

  // ── Derived values ────────────────────────────────────────────────────────

  const equity = account ? parseFloat(account.equity) : null;
  const lastEquity = account ? parseFloat(account.last_equity) : null;
  const todayPL =
    equity != null && lastEquity != null ? equity - lastEquity : null;
  const todayPLPct =
    todayPL != null && lastEquity && lastEquity > 0
      ? todayPL / lastEquity
      : null;
  const buyingPower = account ? parseFloat(account.buying_power) : null;

  const plPositive = todayPL == null || todayPL >= 0;

  const stats = [
    {
      label: "Portfolio Value",
      value: equity != null ? fmtUSD(equity) : "—",
      sub: account?.status === "ACTIVE" ? "Account active" : "—",
      icon: DollarSign,
      iconColor: "text-emerald-400",
      iconBg: "bg-emerald-500/10",
      valueColor: "text-zinc-100",
    },
    {
      label: "Today's P&L",
      value:
        todayPL != null
          ? `${todayPL >= 0 ? "+" : ""}${fmtUSD(todayPL)}`
          : "—",
      sub:
        todayPLPct != null
          ? `${todayPLPct >= 0 ? "+" : ""}${(todayPLPct * 100).toFixed(2)}% vs yesterday`
          : "—",
      icon: TrendingUp,
      iconColor: plPositive ? "text-blue-400" : "text-red-400",
      iconBg: plPositive ? "bg-blue-500/10" : "bg-red-500/10",
      valueColor: plPositive ? "text-zinc-100" : "text-red-400",
    },
    {
      label: "Buying Power",
      value: buyingPower != null ? fmtUSD(buyingPower) : "—",
      sub: "Available to trade",
      icon: Wallet,
      iconColor: "text-indigo-400",
      iconBg: "bg-indigo-500/10",
      valueColor: "text-zinc-100",
    },
    {
      label: "Open Positions",
      value: loading ? "—" : String(positions.length),
      sub:
        positions.length === 0
          ? "No holdings"
          : positions.length === 1
            ? "1 holding"
            : `${positions.length} holdings`,
      icon: Layers,
      iconColor: "text-violet-400",
      iconBg: "bg-violet-500/10",
      valueColor: "text-zinc-100",
    },
  ];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 max-w-7xl">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100">Dashboard</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            {lastUpdated ? `Updated at ${lastUpdated}` : "Paper trading account"}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refresh()}
          disabled={loading}
          className="border-zinc-700 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 gap-2"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Error banner */}
      {error && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4 flex items-center justify-between gap-4">
          <p className="text-sm text-red-400">{error}</p>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refresh()}
            className="text-red-400 hover:text-red-300 shrink-0"
          >
            Retry
          </Button>
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <div
            key={s.label}
            className="bg-zinc-900 border border-zinc-800 rounded-xl p-5"
          >
            <div className="flex items-center justify-between mb-4">
              <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide">
                {s.label}
              </p>
              <div
                className={`w-7 h-7 rounded-lg ${s.iconBg} flex items-center justify-center`}
              >
                <s.icon className={`w-3.5 h-3.5 ${s.iconColor}`} />
              </div>
            </div>
            <p className={cn("text-2xl font-bold tabular-nums", s.valueColor)}>
              {loading && !account ? (
                <span className="text-zinc-700 animate-pulse">···</span>
              ) : (
                s.value
              )}
            </p>
            <p className="text-xs text-zinc-600 mt-1 truncate">{s.sub}</p>
          </div>
        ))}
      </div>

      {/* Equity chart */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-sm font-semibold text-zinc-300">
              Portfolio Performance
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">Equity curve</p>
          </div>
          <div className="flex gap-1">
            {PERIODS.map((p) => (
              <button
                key={p.value}
                onClick={() => handlePeriod(p.value)}
                disabled={chartLoading}
                className={cn(
                  "px-2.5 py-1 text-xs font-medium rounded-md transition-colors",
                  period === p.value
                    ? "bg-indigo-500/20 text-indigo-400"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
                )}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <div className="h-56">
          {chartLoading ? (
            <div className="h-full flex items-center justify-center">
              <p className="text-sm text-zinc-600 animate-pulse">Loading chart…</p>
            </div>
          ) : (
            <EquityChart data={history} baseValue={baseValue} />
          )}
        </div>
      </div>

      {/* Active strategies */}
      {activeStrategies.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-zinc-300 mb-0.5">Active Strategies</h2>
          <p className="text-xs text-zinc-500 mb-4">Strategies currently running against your account</p>
          <div className="space-y-2">
            {activeStrategies.map((s) => (
              <div key={s.id} className="flex items-center gap-3 py-2 border-b border-zinc-800/50 last:border-0">
                <div className={cn(
                  "w-2 h-2 rounded-full shrink-0",
                  s.status === "active" ? "bg-emerald-400" : "bg-amber-400"
                )} />
                <p className="text-sm text-zinc-200 flex-1 font-medium">
                  {s.strategies?.name ?? "Unnamed"}
                </p>
                <Badge variant="outline" className={cn(
                  "text-xs",
                  s.status === "active"
                    ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/5"
                    : "border-amber-500/30 text-amber-400 bg-amber-500/5"
                )}>
                  {s.status === "active" ? (
                    <><Rocket className="w-2.5 h-2.5 mr-1" />Live</>
                  ) : (
                    <><PauseCircle className="w-2.5 h-2.5 mr-1" />Paused</>
                  )}
                </Badge>
                {s.last_run_at && (
                  <p className="text-xs text-zinc-600">
                    Last run {new Date(s.last_run_at).toLocaleDateString()}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Positions table */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-zinc-300 mb-0.5">
          Open Positions
        </h2>
        <p className="text-xs text-zinc-500 mb-5">
          Current holdings in your paper portfolio
        </p>

        {loading && positions.length === 0 ? (
          <div className="h-20 flex items-center justify-center">
            <p className="text-sm text-zinc-600 animate-pulse">
              Loading positions…
            </p>
          </div>
        ) : positions.length === 0 ? (
          <div className="h-20 flex items-center justify-center rounded-lg border border-dashed border-zinc-800">
            <p className="text-sm text-zinc-600">
              No open positions — the bot hasn&apos;t placed any trades yet
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto -mx-1">
            <table className="w-full text-sm min-w-[700px]">
              <thead>
                <tr className="border-b border-zinc-800">
                  {[
                    "Symbol",
                    "Side",
                    "Qty",
                    "Entry",
                    "Current",
                    "Market Value",
                    "Unrealized P&L",
                    "P&L %",
                  ].map((col) => (
                    <th
                      key={col}
                      className="text-left text-xs font-medium text-zinc-500 pb-3 pr-4 last:pr-0 whitespace-nowrap"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {positions.map((p) => {
                  const pl = parseFloat(p.unrealized_pl);
                  const plRatio = parseFloat(p.unrealized_plpc);
                  const pos = pl >= 0;
                  return (
                    <tr
                      key={p.symbol}
                      className="hover:bg-zinc-800/30 transition-colors"
                    >
                      <td className="py-3 pr-4 font-semibold text-zinc-100">
                        {p.symbol}
                      </td>
                      <td className="py-3 pr-4">
                        <Badge
                          variant="outline"
                          className={cn(
                            "text-xs",
                            p.side === "long"
                              ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/5"
                              : "border-red-500/30 text-red-400 bg-red-500/5"
                          )}
                        >
                          {p.side}
                        </Badge>
                      </td>
                      <td className="py-3 pr-4 text-zinc-300 tabular-nums">
                        {fmtQty(p.qty)}
                      </td>
                      <td className="py-3 pr-4 text-zinc-400 tabular-nums">
                        {fmtUSD(p.avg_entry_price)}
                      </td>
                      <td className="py-3 pr-4 text-zinc-300 tabular-nums">
                        {fmtUSD(p.current_price)}
                      </td>
                      <td className="py-3 pr-4 text-zinc-300 tabular-nums">
                        {fmtUSD(p.market_value)}
                      </td>
                      <td
                        className={cn(
                          "py-3 pr-4 tabular-nums font-medium",
                          pos ? "text-emerald-400" : "text-red-400"
                        )}
                      >
                        {`${pos ? "+" : ""}${fmtUSD(pl)}`}
                      </td>
                      <td
                        className={cn(
                          "py-3 tabular-nums font-medium",
                          pos ? "text-emerald-400" : "text-red-400"
                        )}
                      >
                        {fmtPct(plRatio)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
