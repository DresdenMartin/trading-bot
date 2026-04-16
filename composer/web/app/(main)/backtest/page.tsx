"use client";

import { useState, useEffect } from "react";
import { FlaskConical, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { EquityChart } from "@/components/charts/equity-chart";
import type { ChartPoint } from "@/components/charts/equity-chart";
import type { Strategy } from "@/components/strategy-builder/types";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "composer_strategies";

interface BacktestMetrics {
  total_return: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  final_equity: number;
}

interface Trade {
  date: string;
  symbol: string;
  side: "buy" | "sell";
  price: number;
  shares: number;
}

function loadStrategies(): Strategy[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function fmtPct(v: number) {
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

function fmtUSD(v: number) {
  return v.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

export default function BacktestPage() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [startDate, setStartDate] = useState(
    new Date(Date.now() - 365 * 86400_000).toISOString().slice(0, 10)
  );
  const [endDate, setEndDate] = useState(
    new Date().toISOString().slice(0, 10)
  );
  const [capital, setCapital] = useState(100_000);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [equityCurve, setEquityCurve] = useState<ChartPoint[]>([]);
  const [metrics, setMetrics] = useState<BacktestMetrics | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [showBenchmark, setShowBenchmark] = useState(true);

  useEffect(() => {
    setStrategies(loadStrategies());
  }, []);

  async function runBacktest() {
    const strategy = strategies[selectedIdx];
    if (!strategy) return;
    setRunning(true);
    setError(null);
    setEquityCurve([]);
    setMetrics(null);
    setTrades([]);

    try {
      const res = await fetch("/api/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy, start: startDate, end: endDate, capital }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? `Server error ${res.status}`);
        return;
      }

      // Transform equity curve into ChartPoint format
      const baseValue = data.equity_curve[0]?.equity ?? capital;

      // Build benchmark map by date string
      const benchMap: Record<string, number> = {};
      for (const b of (data.benchmark_curve ?? []) as { date: string; equity: number }[]) {
        benchMap[b.date] = b.equity;
      }

      const points: ChartPoint[] = (data.equity_curve as { date: string; equity: number }[]).map(
        (p) => ({
          date: new Date(p.date).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          }),
          equity: p.equity,
          plPct: ((p.equity - baseValue) / baseValue) * 100,
          timestamp: new Date(p.date).getTime() / 1000,
          benchmark: benchMap[p.date] ?? undefined,
        })
      );

      setEquityCurve(points);
      setMetrics(data.metrics);
      setTrades(data.trades ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const metricCards = metrics
    ? [
        {
          label: "Total Return",
          value: fmtPct(metrics.total_return),
          color: metrics.total_return >= 0 ? "text-emerald-400" : "text-red-400",
        },
        {
          label: "Final Equity",
          value: fmtUSD(metrics.final_equity),
          color: "text-zinc-100",
        },
        {
          label: "Sharpe Ratio",
          value: metrics.sharpe.toFixed(2),
          color: metrics.sharpe >= 1 ? "text-emerald-400" : metrics.sharpe >= 0 ? "text-zinc-300" : "text-red-400",
        },
        {
          label: "Max Drawdown",
          value: fmtPct(metrics.max_drawdown),
          color: "text-red-400",
        },
        {
          label: "Win Rate",
          value: `${metrics.win_rate.toFixed(1)}%`,
          color: metrics.win_rate >= 50 ? "text-emerald-400" : "text-zinc-300",
        },
      ]
    : [
        { label: "Total Return", value: "—", color: "text-zinc-500" },
        { label: "Final Equity", value: "—", color: "text-zinc-500" },
        { label: "Sharpe Ratio", value: "—", color: "text-zinc-500" },
        { label: "Max Drawdown", value: "—", color: "text-zinc-500" },
        { label: "Win Rate", value: "—", color: "text-zinc-500" },
      ];

  return (
    <div className="space-y-6 max-w-7xl">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100">Backtest</h1>
        <p className="text-sm text-zinc-500 mt-0.5">
          Replay your strategies against historical market data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Config panel */}
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
          <h2 className="text-sm font-semibold text-zinc-300 mb-5">Configure</h2>
          <div className="space-y-4">
            {/* Strategy picker */}
            <div>
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wide block mb-1.5">
                Strategy
              </label>
              {strategies.length === 0 ? (
                <p className="text-xs text-zinc-600 bg-zinc-800 rounded-lg px-3 py-2">
                  No strategies saved — build one first
                </p>
              ) : (
                <select
                  value={selectedIdx}
                  onChange={(e) => setSelectedIdx(Number(e.target.value))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:border-indigo-500 appearance-none"
                >
                  {strategies.map((s, i) => (
                    <option key={i} value={i}>
                      {s.name}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Date range */}
            <div>
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wide block mb-1.5">
                Date Range
              </label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-indigo-500"
                />
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            {/* Starting capital */}
            <div>
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wide block mb-1.5">
                Starting Capital
              </label>
              <div className="flex items-center bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 gap-1 focus-within:border-indigo-500 transition-colors">
                <span className="text-xs text-zinc-500">$</span>
                <input
                  type="number"
                  value={capital}
                  onChange={(e) => setCapital(Number(e.target.value))}
                  className="flex-1 bg-transparent text-sm text-zinc-300 focus:outline-none"
                />
              </div>
            </div>

            <Button
              onClick={runBacktest}
              disabled={running || strategies.length === 0}
              className="w-full bg-indigo-500 hover:bg-indigo-600 text-white gap-2 text-sm mt-2"
            >
              <Play className={cn("w-3.5 h-3.5", running && "animate-pulse")} />
              {running ? "Running…" : "Run Backtest"}
            </Button>

            {error && (
              <p className="text-xs text-red-400 bg-red-500/5 border border-red-500/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {/* Equity curve */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <div className="flex items-start justify-between mb-1">
              <h2 className="text-sm font-semibold text-zinc-300">Equity Curve</h2>
              {equityCurve.length > 0 && (
                <button
                  onClick={() => setShowBenchmark((v) => !v)}
                  className={cn(
                    "text-xs px-2 py-0.5 rounded-md transition-colors",
                    showBenchmark
                      ? "bg-indigo-500/20 text-indigo-400"
                      : "text-zinc-600 hover:text-zinc-400"
                  )}
                >
                  vs SPY
                </button>
              )}
            </div>
            <p className="text-xs text-zinc-500 mb-4">
              Simulated portfolio value over the backtest period
            </p>
            <div className="h-56">
              {running ? (
                <div className="h-full flex items-center justify-center">
                  <p className="text-sm text-zinc-600 animate-pulse">
                    Running backtest…
                  </p>
                </div>
              ) : equityCurve.length > 0 ? (
                <EquityChart data={equityCurve} baseValue={capital} showBenchmark={showBenchmark} />
              ) : (
                <div className="h-full flex flex-col items-center justify-center gap-3 border border-dashed border-zinc-800 rounded-lg">
                  <FlaskConical className="w-8 h-8 text-zinc-700" />
                  <p className="text-sm text-zinc-600">
                    Run a backtest to see the equity curve
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {metricCards.map((m) => (
              <div
                key={m.label}
                className="bg-zinc-900 border border-zinc-800 rounded-xl p-4"
              >
                <p className="text-xs text-zinc-500 mb-2 leading-tight">
                  {m.label}
                </p>
                <p className={cn("text-lg font-bold tabular-nums", m.color)}>
                  {m.value}
                </p>
              </div>
            ))}
          </div>

          {/* Trade log */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <h2 className="text-sm font-semibold text-zinc-300 mb-1">
              Trade Log
            </h2>
            <p className="text-xs text-zinc-500 mb-4">
              Every simulated order placed
            </p>
            {trades.length === 0 ? (
              <div className="h-20 flex items-center justify-center rounded-lg border border-dashed border-zinc-800">
                <p className="text-sm text-zinc-600">No trades to display</p>
              </div>
            ) : (
              <div className="overflow-x-auto -mx-1">
                <table className="w-full text-xs min-w-[500px]">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      {["Date", "Symbol", "Side", "Price", "Shares"].map(
                        (col) => (
                          <th
                            key={col}
                            className="text-left font-medium text-zinc-500 pb-2 pr-4 last:pr-0"
                          >
                            {col}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800/50">
                    {trades.slice(0, 100).map((t, i) => (
                      <tr key={i} className="hover:bg-zinc-800/30 transition-colors">
                        <td className="py-2 pr-4 text-zinc-400">{t.date}</td>
                        <td className="py-2 pr-4 font-semibold text-zinc-200">
                          {t.symbol}
                        </td>
                        <td className="py-2 pr-4">
                          <span
                            className={cn(
                              "px-1.5 py-0.5 rounded text-xs font-medium",
                              t.side === "buy"
                                ? "bg-emerald-500/10 text-emerald-400"
                                : "bg-red-500/10 text-red-400"
                            )}
                          >
                            {t.side}
                          </span>
                        </td>
                        <td className="py-2 pr-4 text-zinc-300 tabular-nums">
                          ${t.price.toFixed(2)}
                        </td>
                        <td className="py-2 text-zinc-400 tabular-nums">
                          {t.shares.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {trades.length > 100 && (
                  <p className="text-xs text-zinc-600 mt-2 text-center">
                    Showing first 100 of {trades.length} trades
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
