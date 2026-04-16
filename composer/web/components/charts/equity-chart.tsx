"use client";

import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";

export interface ChartPoint {
  date: string;
  equity: number;
  plPct: number | null;
  timestamp: number;
  benchmark?: number;
}

interface EquityChartProps {
  data: ChartPoint[];
  baseValue?: number | null;
  showBenchmark?: boolean;
}

function fmtAxis(val: number): string {
  if (val >= 1_000_000) return `$${(val / 1_000_000).toFixed(2)}M`;
  if (val >= 1_000) return `$${(val / 1_000).toFixed(1)}K`;
  return `$${val.toFixed(0)}`;
}

function fmtFull(val: number): string {
  return val.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const strategyEntry = payload.find((p: { dataKey: string }) => p.dataKey === "equity");
  const benchEntry = payload.find((p: { dataKey: string }) => p.dataKey === "benchmark");
  const equity: number = strategyEntry?.value;
  const bench: number | undefined = benchEntry?.value;
  const plPct: number | null = strategyEntry?.payload?.plPct;
  return (
    <div className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-zinc-400 mb-1">{label}</p>
      <p className="text-zinc-100 font-semibold">{fmtFull(equity)}</p>
      {plPct != null && (
        <p className={plPct >= 0 ? "text-emerald-400" : "text-red-400"}>
          {plPct >= 0 ? "+" : ""}
          {plPct.toFixed(2)}%
        </p>
      )}
      {bench != null && (
        <p className="text-indigo-400 mt-0.5">SPY: {fmtFull(bench)}</p>
      )}
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomLegend({ payload }: any) {
  if (!payload?.length) return null;
  return (
    <div className="flex items-center gap-4 justify-end text-xs text-zinc-500 mt-1 mr-2">
      {payload.map((entry: { color: string; value: string }) => (
        <div key={entry.value} className="flex items-center gap-1.5">
          <div
            className="w-5 h-0.5"
            style={{ background: entry.color }}
          />
          <span>{entry.value}</span>
        </div>
      ))}
    </div>
  );
}

export function EquityChart({ data, baseValue, showBenchmark }: EquityChartProps) {
  if (!data.length) {
    return (
      <div className="h-full flex items-center justify-center rounded-lg border border-dashed border-zinc-800">
        <p className="text-sm text-zinc-600">
          No history data for this period yet
        </p>
      </div>
    );
  }

  const last = data[data.length - 1].equity;
  const base = baseValue ?? data[0].equity;
  const isPositive = last >= base;
  const color = isPositive ? "#10b981" : "#ef4444";
  const hasBenchmark = showBenchmark && data.some((d) => d.benchmark != null);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.15} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />

        <XAxis
          dataKey="date"
          tick={{ fill: "#71717a", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
          minTickGap={64}
        />

        <YAxis
          tickFormatter={fmtAxis}
          tick={{ fill: "#71717a", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={64}
          domain={["auto", "auto"]}
        />

        {base != null && (
          <ReferenceLine
            y={base}
            stroke="#3f3f46"
            strokeDasharray="4 3"
            strokeWidth={1}
          />
        )}

        <Tooltip content={<CustomTooltip />} />

        {hasBenchmark && <Legend content={<CustomLegend />} />}

        <Area
          type="monotone"
          dataKey="equity"
          name="Strategy"
          stroke={color}
          strokeWidth={1.5}
          fill="url(#equityGrad)"
          dot={false}
          activeDot={{ r: 3, fill: color, strokeWidth: 0 }}
        />

        {hasBenchmark && (
          <Line
            type="monotone"
            dataKey="benchmark"
            name="SPY"
            stroke="#6366f1"
            strokeWidth={1}
            strokeDasharray="4 3"
            dot={false}
            activeDot={{ r: 2, fill: "#6366f1", strokeWidth: 0 }}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
