import { NextResponse } from "next/server";
import { getUserAlpaca } from "@/lib/get-user-alpaca";

// Maps period → sensible default timeframe
const TIMEFRAME_MAP: Record<string, string> = {
  "1D": "5Min",
  "1W": "1H",
  "1M": "1D",
  "3M": "1D",
  "6M": "1D",
  "1A": "1D",
};

function formatDate(ts: number, period: string): string {
  const d = new Date(ts * 1000);
  if (period === "1D") {
    return d.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
      timeZone: "America/New_York",
    });
  }
  if (period === "1W") {
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      timeZone: "America/New_York",
    });
  }
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "America/New_York",
  });
}

export async function GET(request: Request) {
  const creds = await getUserAlpaca();
  if (!creds) {
    return NextResponse.json(
      { error: "Alpaca keys not configured. Go to Settings to add yours." },
      { status: 503 }
    );
  }

  const { searchParams } = new URL(request.url);
  const period = searchParams.get("period") ?? "1M";
  const timeframe =
    searchParams.get("timeframe") ?? TIMEFRAME_MAP[period] ?? "1D";

  try {
    const url = new URL(`${creds.base}/v2/account/portfolio/history`);
    url.searchParams.set("period", period);
    url.searchParams.set("timeframe", timeframe);
    url.searchParams.set("extended_hours", "false");

    const res = await fetch(url.toString(), {
      headers: {
        "APCA-API-KEY-ID": creds.key,
        "APCA-API-SECRET-KEY": creds.secret,
      },
      cache: "no-store",
    });

    if (!res.ok) {
      const detail = await res.text();
      return NextResponse.json(
        { error: `Alpaca API returned ${res.status}`, detail },
        { status: res.status }
      );
    }

    const raw = await res.json();

    const timestamps: number[] = raw.timestamp ?? [];
    const equities: (number | null)[] = raw.equity ?? [];
    const plPcts: (number | null)[] = raw.profit_loss_pct ?? [];

    // Build chart-friendly points, dropping nulls
    const points = timestamps
      .map((ts, i) => ({
        date: formatDate(ts, period),
        equity: equities[i],
        plPct: plPcts[i] != null ? (plPcts[i] as number) * 100 : null,
        timestamp: ts,
      }))
      .filter((p): p is {
        date: string;
        equity: number;
        plPct: number | null;
        timestamp: number;
      } => p.equity != null);

    return NextResponse.json({
      points,
      base_value: raw.base_value ?? null,
      timeframe: raw.timeframe ?? timeframe,
      period,
    });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to reach Alpaca API", detail: String(err) },
      { status: 500 }
    );
  }
}
