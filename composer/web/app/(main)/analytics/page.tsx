"use client";

import { useState, useEffect, useCallback } from "react";
import { RefreshCw, TrendingUp, TrendingDown, AlertTriangle, BarChart2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
} from "recharts";
import { cn } from "@/lib/utils";

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

interface AlpacaAccount {
  equity: string;
  cash: string;
  buying_power: string;
}

// Sector mapping for common symbols
const SECTOR_MAP: Record<string, string> = {
  AAPL: "Technology", MSFT: "Technology", NVDA: "Technology",
  AMD: "Technology", GOOGL: "Technology", GOOG: "Technology",
  META: "Technology", AMZN: "Consumer", TSLA: "Automotive",
  SPY: "Index", QQQ: "Index", HOOD: "Financials",
  SOFI: "Financials", COIN: "Crypto", PLTR: "Technology",
};

const CHART_COLORS = [
  "#6366f1", "#8b5cf6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#ec4899", "#14b8a6",
];

function fmtUSD(v: number): string {
  const abs = Math.abs(v);
  return `${v < 0 ? "-" : ""}${abs.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function fmtPct(v: number): string {
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function PieTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const { name, value, pct } = payload[0].payload;
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="font-semibold text-zinc-100">{name}</p>
      <p className="text-zinc-300">{fmtUSD(value)}</p>
      <p className="text-zinc-500">{pct.toFixed(1)}%</p>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function BarTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const val: number = payload[0].value;
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="font-semibold text-zinc-100">{label}</p>
      <p className={val >= 0 ? "text-emerald-400" : "text-red-400"}>{fmtUSD(val)}</p>
    </div>
  );
}

export default function AnalyticsPage() {
  const [positions, setPositions] = useState<AlpacaPosition[]>([]);
  const [account, setAccount] = useState<AlpacaAccount | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [posRes, acctRes] = await Promise.all([
        fetch("/api/alpaca/positions"),
        fetch("/api/alpaca/account"),
      ]);
      if (!acctRes.ok) throw new Error("Failed to load account");
      const [pos, acct] = await Promise.all([
        posRes.ok ? posRes.json() : [],
        acctRes.json(),
      ]);
      setPositions(Array.isArray(pos) ? pos : []);
      setAccount(acct);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  // ── Derived data ─────────────────────────────────────────────────────────────

  const totalEquity = account ? parseFloat(account.equity) : 0;
  const cash = account ? parseFloat(account.cash) : 0;
  const investedValue = positions.reduce((s, p) => s + parseFloat(p.market_value), 0);
  const totalUnrealizedPL = positions.reduce((s, p) => s + parseFloat(p.unrealized_pl), 0);

  // Allocation pie data
  const allocationData = [
    ...positions.map((p, i) => ({
      name: p.symbol,
      value: parseFloat(p.market_value),
      pct: (parseFloat(p.market_value) / totalEquity) * 100,
      color: CHART_COLORS[i % CHART_COLORS.length],
    })),
    ...(cash > 0
      ? [{ name: "Cash", value: cash, pct: (cash / totalEquity) * 100, color: "#3f3f46" }]
      : []),
  ].sort((a, b) => b.value - a.value);

  // Sector breakdown
  const sectorMap: Record<string, number> = {};
  for (const p of positions) {
    const sector = SECTOR_MAP[p.symbol] ?? "Other";
    sectorMap[sector] = (sectorMap[sector] ?? 0) + parseFloat(p.market_value);
  }
  const sectorData = Object.entries(sectorMap)
    .map(([name, value], i) => ({
      name,
      value,
      pct: (value / totalEquity) * 100,
      color: CHART_COLORS[i % CHART_COLORS.length],
    }))
    .sort((a, b) => b.value - a.value);

  // P&L bar chart
  const plData = positions
    .map((p) => ({
      symbol: p.symbol,
      pl: parseFloat(p.unrealized_pl),
      plPct: parseFloat(p.unrealized_plpc) * 100,
    }))
    .sort((a, b) => b.pl - a.pl);

  // Best/worst
  const best = plData[0];
  const worst = plData[plData.length - 1];
  const largestPos = allocationData.filter((d) => d.name !== "Cash")[0];
  const concentrationRisk = largestPos ? largestPos.pct : 0;

  if (loading) {
    return (
      <div className="space-y-4 max-w-7xl">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-zinc-100">Analytics</h1>
            <p className="text-sm text-zinc-500 mt-0.5">Portfolio breakdown and risk analysis</p>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-xl h-72 animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100">Analytics</h1>
          <p className="text-sm text-zinc-500 mt-0.5">Portfolio breakdown and risk analysis</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchData}
          disabled={loading}
          className="border-zinc-700 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 gap-2"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {positions.length === 0 && !error ? (
        <div className="bg-zinc-900 border border-dashed border-zinc-800 rounded-xl p-16 flex flex-col items-center gap-4">
          <BarChart2 className="w-10 h-10 text-zinc-700" />
          <div className="text-center">
            <p className="text-sm font-medium text-zinc-400">No positions to analyze</p>
            <p className="text-xs text-zinc-600 mt-1">Place some trades first to see portfolio analytics</p>
          </div>
        </div>
      ) : (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                label: "Total Invested",
                value: fmtUSD(investedValue),
                sub: `${((investedValue / totalEquity) * 100).toFixed(1)}% of portfolio`,
                color: "text-zinc-100",
              },
              {
                label: "Total Unrealized P&L",
                value: fmtUSD(totalUnrealizedPL),
                sub: fmtPct((totalUnrealizedPL / (investedValue || 1)) * 100),
                color: totalUnrealizedPL >= 0 ? "text-emerald-400" : "text-red-400",
              },
              {
                label: "Best Position",
                value: best ? best.symbol : "—",
                sub: best ? `${fmtUSD(best.pl)} (${fmtPct(best.plPct)})` : "—",
                color: "text-emerald-400",
                icon: TrendingUp,
              },
              {
                label: "Worst Position",
                value: worst && worst.pl < 0 ? worst.symbol : "—",
                sub: worst && worst.pl < 0 ? `${fmtUSD(worst.pl)} (${fmtPct(worst.plPct)})` : "—",
                color: worst && worst.pl < 0 ? "text-red-400" : "text-zinc-400",
                icon: TrendingDown,
              },
            ].map((s) => (
              <div key={s.label} className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
                <p className="text-xs font-medium text-zinc-500 uppercase tracking-wide mb-3">{s.label}</p>
                <p className={cn("text-xl font-bold tabular-nums", s.color)}>{s.value}</p>
                <p className="text-xs text-zinc-500 mt-1">{s.sub}</p>
              </div>
            ))}
          </div>

          {/* Concentration warning */}
          {concentrationRisk > 40 && (
            <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 flex items-center gap-3">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <p className="text-sm text-amber-400">
                <span className="font-semibold">{largestPos.name}</span> represents{" "}
                {concentrationRisk.toFixed(1)}% of your portfolio — consider diversifying.
              </p>
            </div>
          )}

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Allocation pie */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-zinc-300 mb-1">Allocation</h2>
              <p className="text-xs text-zinc-500 mb-4">Portfolio weight by market value</p>
              <div className="flex items-center gap-4">
                <div className="w-44 h-44 shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={allocationData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={72}
                        dataKey="value"
                        paddingAngle={2}
                      >
                        {allocationData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} stroke="transparent" />
                        ))}
                      </Pie>
                      <Tooltip content={<PieTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex-1 space-y-1.5 min-w-0">
                  {allocationData.map((entry) => (
                    <div key={entry.name} className="flex items-center gap-2 text-xs">
                      <div
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ background: entry.color }}
                      />
                      <span className="font-medium text-zinc-200 w-12 truncate">{entry.name}</span>
                      <span className="text-zinc-500 tabular-nums ml-auto">{entry.pct.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Sector pie */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-zinc-300 mb-1">Sector Exposure</h2>
              <p className="text-xs text-zinc-500 mb-4">Holdings grouped by sector</p>
              <div className="flex items-center gap-4">
                <div className="w-44 h-44 shrink-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={sectorData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={72}
                        dataKey="value"
                        paddingAngle={2}
                      >
                        {sectorData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} stroke="transparent" />
                        ))}
                      </Pie>
                      <Tooltip content={<PieTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex-1 space-y-1.5 min-w-0">
                  {sectorData.map((entry) => (
                    <div key={entry.name} className="flex items-center gap-2 text-xs">
                      <div
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ background: entry.color }}
                      />
                      <span className="font-medium text-zinc-200 truncate flex-1">{entry.name}</span>
                      <span className="text-zinc-500 tabular-nums">{entry.pct.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* P&L bar chart */}
          {plData.length > 0 && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-zinc-300 mb-1">Unrealized P&L by Position</h2>
              <p className="text-xs text-zinc-500 mb-4">Dollar gain/loss per holding</p>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={plData}
                    margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis
                      dataKey="symbol"
                      tick={{ fill: "#71717a", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      tickFormatter={(v) => `$${v >= 0 ? "" : "-"}${Math.abs(v) >= 1000 ? `${(Math.abs(v) / 1000).toFixed(1)}K` : Math.abs(v).toFixed(0)}`}
                      tick={{ fill: "#71717a", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={60}
                    />
                    <ReferenceLine y={0} stroke="#3f3f46" strokeWidth={1} />
                    <Tooltip content={<BarTooltip />} />
                    <Bar
                      dataKey="pl"
                      radius={[3, 3, 0, 0]}
                    >
                      {plData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.pl >= 0 ? "#10b981" : "#ef4444"}
                          opacity={0.85}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Position detail table */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5">
            <h2 className="text-sm font-semibold text-zinc-300 mb-1">Position Detail</h2>
            <p className="text-xs text-zinc-500 mb-4">Full breakdown by holding</p>
            <div className="overflow-x-auto -mx-1">
              <table className="w-full text-xs min-w-[600px]">
                <thead>
                  <tr className="border-b border-zinc-800">
                    {["Symbol", "Sector", "Market Value", "Weight", "Unrealized P&L", "P&L %", "Today"].map((col) => (
                      <th key={col} className="text-left font-medium text-zinc-500 pb-2 pr-4 last:pr-0">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50">
                  {positions.map((p) => {
                    const pl = parseFloat(p.unrealized_pl);
                    const plPct = parseFloat(p.unrealized_plpc) * 100;
                    const mv = parseFloat(p.market_value);
                    const weight = (mv / totalEquity) * 100;
                    const today = parseFloat(p.change_today) * 100;
                    return (
                      <tr key={p.symbol} className="hover:bg-zinc-800/30 transition-colors">
                        <td className="py-2.5 pr-4 font-semibold text-zinc-100">{p.symbol}</td>
                        <td className="py-2.5 pr-4 text-zinc-500">{SECTOR_MAP[p.symbol] ?? "Other"}</td>
                        <td className="py-2.5 pr-4 text-zinc-300 tabular-nums">{fmtUSD(mv)}</td>
                        <td className="py-2.5 pr-4 text-zinc-400 tabular-nums">{weight.toFixed(1)}%</td>
                        <td className={cn("py-2.5 pr-4 tabular-nums font-medium", pl >= 0 ? "text-emerald-400" : "text-red-400")}>
                          {fmtUSD(pl)}
                        </td>
                        <td className={cn("py-2.5 pr-4 tabular-nums", pl >= 0 ? "text-emerald-400" : "text-red-400")}>
                          {fmtPct(plPct)}
                        </td>
                        <td className={cn("py-2.5 tabular-nums", today >= 0 ? "text-zinc-300" : "text-red-400")}>
                          {fmtPct(today)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
